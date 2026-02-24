from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict

app = FastAPI(title="CheckMe Risk Triage API")

class InputData(BaseModel):
    age: int
    age_group: str
    family_history: int
    previous_lumps: int
    breast_pain: int
    nipple_discharge: int
    skin_dimples: int
    lump_size_mm: float
    symptom_duration_days: int
    pregnancy_status: int
    hormonal_contraception: int
    fever: int
    weight_loss: int
    fatigue: int
    region: str
    language: str

@app.post("/predict")
def run_prediction(data: InputData):
    return predict(data.dict())
