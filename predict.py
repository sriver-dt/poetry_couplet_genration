import json
import logging
import os.path
import os
import sys
import warnings

dir_name = os.path.dirname(__file__)
sys.path.append(dir_name)

import torch
import transformers

from net.custom_transformer import Transformer
from net.mt5_net import MT5BaseNet
from net.gpt2_net import GPT2BaseNet
from transformers import AutoConfig, GPT2Config, GPT2Tokenizer
from tokenizer.tokenizer import PegasusTokenizer

warnings.filterwarnings('ignore', category=UserWarning)
transformers.logging.set_verbosity_error()


def score_processor(scores: torch.Tensor, temperature: float, top_k: int):
    scores = scores / temperature
    top_k = min(top_k, scores.size(-1))
    # 获取小于top_k中最小置信度的位置索引
    indices_to_remove = scores < torch.topk(scores, top_k, dim=-1).values[..., -1:]
    # 将非top_k位置的置信度置为负无穷
    scores = scores.masked_fill(indices_to_remove, torch.finfo(torch.float32).min)
    return scores


class CustomPredictor:
    def __init__(self, model_dir, do_sample=True, temperature=1.0, top_k=5):
        with open(os.path.join(model_dir, 'vocab.json'), 'r', encoding='utf-8') as reader:
            self.token2idx = json.load(reader)
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k

        # 模型参数恢复
        with open(os.path.join(model_dir, 'config.json'), 'r', encoding='utf-8') as reader:
            self.config = json.load(reader)
        self.config['is_training'] = False
        self.config['dropout'] = 0.0

        net_states = torch.load(os.path.join(model_dir, 'best.pkl'), map_location=torch.device('cpu'))['model_state']
        self.net = Transformer(**self.config)
        logging.info('正在进行模型参数恢复')
        missing_keys, unexpected_keys = self.net.load_state_dict(state_dict=net_states, strict=False)
        logging.info(f'未恢复的参数:{missing_keys}')
        logging.info(f'未解析的参数:{unexpected_keys}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def predict(self, text: str, max_len=20):
        pred_tokens = []
        self.net.eval().to(device=self.device)
        enc_input_ids = torch.tensor(self.convert_token_to_id(text), dtype=torch.int64, device=self.device)[None, ...]
        dec_input_ids = [[self.token2idx['<START>']]]
        dec_input_ids = torch.tensor(dec_input_ids, dtype=torch.long, device=self.device)
        while True:
            output = self.net(input_ids=enc_input_ids, decoder_input_ids=dec_input_ids)  # [N, T, vocab_size]
            next_token_score = output.clone()[:, -1, :].float().cpu()
            next_token_score = score_processor(scores=next_token_score, temperature=self.temperature, top_k=self.top_k)
            if self.do_sample:
                # sample
                probs = torch.nn.functional.softmax(next_token_score, dim=-1)
                # 按照置信度重置后的softmax概率分布进行抽样
                pred_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
            else:
                # greedy search
                pred_id = torch.argmax(next_token_score, dim=-1)  # [1, 1]

            dec_input_ids = torch.concatenate(
                [dec_input_ids, pred_id.to(dtype=dec_input_ids.dtype, device=dec_input_ids.device)], dim=-1)

            pred_token = self.idx2token[pred_id[0].item()]
            if pred_token == '<END>' or len(pred_tokens) >= max_len:
                return ''.join(pred_tokens)
            pred_tokens.append(pred_token)

    def convert_token_to_id(self, text):
        ids = [self.token2idx[token] if token in self.token2idx.keys() else self.token2idx['<UNK>'] for token in text]
        ids.insert(0, self.token2idx['<START>'])
        ids.append(self.token2idx['<END>'])
        return ids


class T5Predictor:
    def __init__(self, t5_model_path, model_dir, do_sample=True, temperature=1.0, top_k=5):
        self.t5_tokenizer = PegasusTokenizer.from_pretrained(t5_model_path, legacy=False)
        self.t5_config = AutoConfig.from_pretrained(t5_model_path)
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k

        # 模型参数恢复
        net_states = torch.load(os.path.join(model_dir, 'last.pkl'), map_location=torch.device('cpu'))['model_state']
        self.net = MT5BaseNet(model_path=t5_model_path)
        logging.info('正在进行模型参数恢复')
        missing_keys, unexpected_keys = self.net.load_state_dict(state_dict=net_states, strict=False)
        logging.info(f'未恢复的参数:{missing_keys}')
        logging.info(f'未解析的参数:{unexpected_keys}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def predict(self, text: str):
        pred_tokens = []
        self.net.eval().to(device=self.device)
        input_ids = self.t5_tokenizer.convert_tokens_to_ids(list(text))
        input_ids.append(self.t5_tokenizer.vocab[self.t5_tokenizer.eos_token])
        dec_input_ids = [self.t5_config.decoder_start_token_id]
        while True:
            output = self.net(
                input_ids=torch.tensor(input_ids, dtype=torch.int64, device=self.device)[None, ...],
                decoder_input_ids=torch.tensor(dec_input_ids, dtype=torch.int64, device=self.device)[None, ...]
            )
            next_token_score = output.clone()[:, -1, :].float().cpu()
            next_token_score = score_processor(scores=next_token_score, temperature=self.temperature, top_k=self.top_k)
            if self.do_sample:
                # sample
                probs = torch.nn.functional.softmax(next_token_score, dim=-1)
                # 按照置信度重置后的softmax概率分布进行抽样
                pred_id = torch.multinomial(probs, num_samples=1).item()
            else:
                # greedy search
                pred_id = torch.argmax(next_token_score[-1], dim=-1).item()
            dec_input_ids.append(pred_id)
            pred_token = self.t5_tokenizer.convert_ids_to_tokens(pred_id)
            if pred_token == self.t5_tokenizer.eos_token or len(pred_tokens) >= 20:
                break
            pred_tokens.append(pred_token)
        return ''.join(pred_tokens)


class GPT2Predictor:
    def __init__(self, gpt2_model_path, model_dir, do_sample=True, temperature=1.0, top_k=5):
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path, legacy=False)
        self.gpt2_config = GPT2Config.from_pretrained(gpt2_model_path)
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k

        # 模型参数恢复
        net_states = torch.load(os.path.join(model_dir, 'best.pkl'), map_location=torch.device('cpu'))['model_state']
        self.net = GPT2BaseNet(model_path=gpt2_model_path)
        logging.info('正在进行模型参数恢复')
        missing_keys, unexpected_keys = self.net.load_state_dict(state_dict=net_states, strict=False)
        logging.info(f'未恢复的参数:{missing_keys}')
        logging.info(f'未解析的参数:{unexpected_keys}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def predict(self, text: str):
        pred_ids = []
        self.net.eval().to(device=self.device)
        input_ids = self.gpt2_tokenizer(text + ' ')['input_ids']
        while True:
            output = self.net(
                input_ids=torch.tensor(input_ids, dtype=torch.int64, device=self.device)[None, ...],
            )
            next_token_score = output.clone()[:, -1, :].float().cpu()
            next_token_score = score_processor(scores=next_token_score, temperature=self.temperature, top_k=self.top_k)
            if self.do_sample:
                # sample
                probs = torch.nn.functional.softmax(next_token_score, dim=-1)
                # 按照置信度重置后的softmax概率分布进行抽样
                pred_id = torch.multinomial(probs, num_samples=1).item()
            else:
                # greedy search
                pred_id = torch.argmax(next_token_score[-1], dim=-1).item()
            input_ids.append(pred_id)
            if pred_id == self.gpt2_tokenizer.eos_token_id or len(pred_ids) >= 200:
                break
            pred_ids.append(pred_id)
        pred_text = self.gpt2_tokenizer.decode(pred_ids)
        return pred_text


if __name__ == '__main__':
    do_sample_ = True
    temperature_ = 1.2
    top_k_ = 10

    predictor = CustomPredictor('./output/custom_transformer', do_sample_, temperature_, top_k_)
    print('CustomPredictor: ')
    print('请根据给定的上联生成下联：冰壶见底未为清')
    print('下联：', predictor.predict('请根据给定的上联生成下联：冰壶见底未为清'))
    print('-' * 50)
    print('请根据给定的上下联第一个字生成完整对联：天地')
    print('上下联：', predictor.predict('请根据给定的上下联第一个字生成完整对联：天地'))

    t5predictor = T5Predictor(
        r'D:\huggingface\hub\t5-pegasus-small', './output/t5-pegasus',
        do_sample_, temperature_, top_k_
    )
    print('MT5: ')
    print('请根据给定的上联生成下联：冰壶见底未为清')
    print('下联：', t5predictor.predict('请根据给定的上联生成下联：冰壶见底未为清'))
    print('-' * 50)
    print('请根据给定的上下联第一个字生成完整对联：天地')
    print('上下联：', t5predictor.predict('请根据给定的上下联第一个字生成完整对联：天地'))

    gpt2_predictor = GPT2Predictor(
        r'D:\huggingface\hub\gpt2', os.path.join(dir_name, './output/gpt2'),
        do_sample_, temperature_, top_k_
    )
    print('GPT2: ')
    print('请根据给定的上联生成下联，上联：嘈嘈切切错杂弹')
    print(gpt2_predictor.predict('请根据给定的上联生成下联，上联：嘈嘈切切错杂弹'))
    print('-' * 50)
    print('请根据给定的上下联第一个字生成完整对联，上下联第一个字：春秋')
    print(gpt2_predictor.predict('请根据给定的上下联第一个字生成完整对联，上下联第一个字：春秋'))
