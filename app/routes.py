from fastapi import APIRouter
from .model_loader import model
from .schemas import PredictRequest, PredictResponse

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    # If real model is not present → dummy output
    if model is None:
        return {"clickbait_score": 0.5}

    # When real model is added, update this part
    score = 0.7  # temporary
    return {"clickbait_score": score}
