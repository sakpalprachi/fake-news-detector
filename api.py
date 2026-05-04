import requests
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from ml_model import FakeNewsDetector

app = FastAPI(title="Fake News Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = FakeNewsDetector()

# ================= MODELS =================
class NewsRequest(BaseModel):
    text: str
    title: Optional[str] = None


# ================= STARTUP =================
@app.on_event("startup")
async def startup():
    if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        detector.load_model()
        print("Model Loaded ✅")
    else:
        print("Training Model...")
        detector.train()
        detector.save_model()


# ================= BASIC =================
@app.get("/")
def home():
    return {"message": "API Running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok", "model_trained": detector.is_trained}


# ================= PREDICT =================
@app.post("/predict")
def predict(req: NewsRequest):
    if not detector.is_trained:
        raise HTTPException(status_code=500, detail="Model not ready")

    text = req.text
    if req.title:
        text = req.title + " " + req.text

    result = detector.predict(text)

    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
    }


# ================= LIVE NEWS =================
@app.get("/live-news")
def live_news():
    API_KEY = "4edeabe8fb624eafbff6d11d0278aa21"

    # 🔥 FIX: country changed to India
    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={API_KEY}"

    try:
        res = requests.get(url)
        data = res.json()

        articles = []
        for article in data.get("articles", []):
            articles.append({
                "title": article.get("title"),
                "description": article.get("description")
            })

        return {"status": "success", "news": articles}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ================= FACT CHECK =================
@app.get("/fact-check")
def fact_check(query: str):
    try:
        # 🔥 FIX: better query + language filter
        query = "latest news " + query

        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&token=b0935ad85afdffba1d484751039992a4"

        res = requests.get(url)
        data = res.json()

        results = []
        for article in data.get("articles", [])[:5]:
            results.append({
                "title": article.get("title"),
                "source": article.get("source", {}).get("name"),
                "url": article.get("url")
            })

        # 🔥 FIX: added 'found' key
        return {
            "status": "success",
            "found": len(results) > 0,
            "results": results
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ================= RUN =================
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)