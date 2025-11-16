# from pydantic import BaseModel

# class PredictRequest(BaseModel):
#     text: str

# class PredictResponse(BaseModel):
#     clickbait_score: float

# New (Below)

from pydantic import BaseModel

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    clickbait_prediction: str
    clickbait_score: float
    emotion_prediction: str
    emotion_score: float
    model_status: str