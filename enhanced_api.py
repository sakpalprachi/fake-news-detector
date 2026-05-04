from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB

# ---------------- CONFIG ----------------
NEWS_API_KEY = "PASTE_YOUR_API_KEY_HERE"

app = FastAPI(title="Enhanced Fake News Detector API")

# ---------------- INPUT MODEL ----------------
class NewsInput(BaseModel):
    title: str = ""
    content: str

# ---------------- SAMPLE TRAIN DATA ----------------
texts = [
    "Government launches new policy for education reform",
    "Scientists discover new species in Amazon rainforest",
    "NASA successfully launches satellite into orbit",
    "Aliens landed in India and government is hiding truth",
    "Drinking hot water cures all diseases instantly",
    "Celebrity confirms earth is flat in interview"
]

labels = [1, 1, 1, 0, 0, 0]  # 1=Real, 0=Fake

# ---------------- VECTORIZER ----------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(texts)

# ---------------- MODELS ----------------
lr = LogisticRegression()
rf = RandomForestClassifier()
nb = MultinomialNB()

model = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('nb', nb)
    ],
    voting='soft'
)

model.fit(X, labels)

# ---------------- ROUTES ----------------

@app.get("/")
def home():
    return {"message": "API running successfully"}

# 🔹 Predict
@app.post("/predict")
def predict(news: NewsInput):
    try:
        text = news.title + " " + news.content
        vec = vectorizer.transform([text])

        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        confidence = max(prob) * 100

        return {
            "prediction": "Real News" if pred == 1 else "Fake News",
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}

# 🔹 Explain
@app.post("/explain")
def explain(news: NewsInput):
    return {
        "explanation": "The model uses multiple ML algorithms (Logistic Regression, Random Forest, Naive Bayes) and TF-IDF features to analyze patterns and classify the news."
    }

# 🔹 Summarize
@app.post("/summarize")
def summarize(news: NewsInput):
    try:
        text = news.content
        summary = " ".join(text.split(".")[:2])

        return {
            "summary": summary if summary else "No content provided"
        }

    except Exception as e:
        return {"error": str(e)}

# 🔹 LIVE NEWS (REAL API)
@app.get("/live-news")
def live_news():
    url = f"https://newsapi.org/v2/top-headlines?country=in&pageSize=5&apiKey={NEWS_API_KEY}"

    try:
        response = requests.get(url)
        data = response.json()

        if data.get("status") != "ok":
            return {"error": "Failed to fetch news"}

        articles = data.get("articles", [])

        news_list = []
        for article in articles:
            news_list.append({
                "title": article.get("title"),
                "description": article.get("description")
            })

        return {"news": news_list}

    except Exception as e:
        return {"error": str(e)}

# ---------------- RUN ----------------
if __name__ == "__main__":
    uvicorn.run("enhanced_api:app", host="127.0.0.1", port=8000, reload=True)