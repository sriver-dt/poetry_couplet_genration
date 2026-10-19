import uvicorn
from fastapi import FastAPI, Body

from predict import CustomPredictor

app = FastAPI()
predictor = CustomPredictor('../output/custom_transformer')


# @app.get("/")
# def read_root():
#     return "使用fastapi框架搭建诗词对联生成的模型预测"


async def predict(text: str = Body(..., description="待预测文本text")):
    print("调用predict方法......")
    pred_result = predictor.predict(text=text)
    # 拼接返回最终值
    return {'code': 0, 'msg': '成功', 'data': pred_result, 'text': text}


app.post("/predict", summary="Predict预测接口")(predict)

if __name__ == '__main__':
    # http://127.0.0.1:8001/docs
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
