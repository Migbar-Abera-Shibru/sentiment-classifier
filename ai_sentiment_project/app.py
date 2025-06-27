from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/tfidf.pkl')

app = FastAPI()

class ReviewRequest(BaseModel):
    review: str

@app.post("/predict")
async def predict_sentiment(data: ReviewRequest):
    X = vectorizer.transform([data.review])
    probs = model.predict_proba(X)[0]
    label = model.predict(X)[0]
    sentiment = "positive" if label == 1 else "negative"
    return {
        "sentiment": sentiment,
        "confidence": round(float(max(probs)), 2)
    }
