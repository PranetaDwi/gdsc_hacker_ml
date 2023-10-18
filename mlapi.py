from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")

app = FastAPI()

class ScoringItem(BaseModel):
    usia : float
    gender: int
    sistol : float
    diastol : float
    nadi : float 
    tb : float
    bb : float

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get("/")
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/submit")
async def scoring_endpoint(request: Request, item:ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return templates.TemplateResponse("result.html", {"request":request, "prediction": yhat[0]})

