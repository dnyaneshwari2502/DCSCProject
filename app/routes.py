# from fastapi import APIRouter
# from .model_loader import model
# from .schemas import PredictRequest, PredictResponse

# router = APIRouter()

# @router.post("/predict", response_model=PredictResponse)
# def predict(data: PredictRequest):
#     # If real model is not present → dummy output
#     if model is None:
#         return {"clickbait_score": 0.5}

#     # When real model is added, update this part
#     score = 0.7  # temporary
#     return {"clickbait_score": score}

# NEW (below)

from fastapi import APIRouter
from .schemas import PredictRequest, PredictResponse
from .service import predict_clickbait_and_emotion  # Import the function

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    # Call the prediction function from the service
    prediction_results = predict_clickbait_and_emotion(data.text)
    return prediction_results