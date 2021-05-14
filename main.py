##################################################
## Author: Jefferson Rivera
## Email: riverajefer@gmail.com
##################################################

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from models.ModelsPredict import ModelPredict
from models.MetricsModel import MetricsModelPredict, PreparateData


METRICSMODELPREDICT = MetricsModelPredict()
MODEL_PREDICT = ModelPredict()
PREPARATE_DATA = PreparateData()

split = PREPARATE_DATA.onSplit()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class NameModel(BaseModel):
    name_model: str

class PredictText(BaseModel):
    text: str


def getPreditcTextDic(text):
    data = [
        {
            'model': 'LogisticRegression',
            'emotion': MODEL_PREDICT.onPredictLogisticRegression(text=text),
            'score': 0.87
        },
        {
            'model': 'DecisionTreeClassifier',
            'emotion': MODEL_PREDICT.onPredictDecisionTreeClassifier(text=text),
            'score': 0.87
        },
        {
            'model': 'MLPClassifier',
            'emotion': MODEL_PREDICT.onPredictMLPClassifierPredic(text=text),
            'score': 0.87
        },
        {
            'model': 'RandomForestClassifier',
            'emotion': MODEL_PREDICT.onRandomForestClassifier(text=text),
            'score': 0.87
        },
    ]
    
    return data


@app.get("/", response_class=HTMLResponse)
def read_root_test(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
def main(pred: PredictText):
    return getPreditcTextDic(text=pred.text)


@app.post("/get_metrics_by_model")
def get_metrics(nameModel: NameModel):
    print(nameModel.name_model)
    metrics = METRICSMODELPREDICT.onGetMetricsModel(nameModel=nameModel.name_model, split=split)

    return metrics

