# 📰 AI & ML Fake News Detection System

## 🚀 Overview
This project is an **AI and Machine Learning based Fake News Detection System** that classifies news articles as **REAL or FAKE** using Natural Language Processing (NLP) and multiple ML models. It also includes **AI-powered explanations, summaries, and fact-checking suggestions** for better understanding.

---

## 🎯 Objective
- Detect fake news using Machine Learning  
- Improve accuracy using multiple ML models  
- Use ensemble learning for final prediction  
- Provide confidence score for results  
- Add AI-based explanation and summarization  
- Build full-stack system using FastAPI + Streamlit  

---

## 🧠 Features

### 🤖 Machine Learning
- Logistic Regression  
- Passive Aggressive Classifier  
- Multinomial Naive Bayes  
- Random Forest  
- Voting Classifier (Ensemble Model)  
- TF-IDF Vectorization (n-grams 1,2)

---

### 📊 Prediction Output
- Real / Fake classification  
- Confidence score (e.g., 89% Fake)  

---

### 🧠 AI Features
- Explain why news is fake or real  
- Summarize news in 3–4 lines  
- Fact-check suggestions  

---

### 🌐 System Features
- FastAPI backend  
- Streamlit frontend UI  
- REST API integration  

---

## ⚙️ Tech Stack
Python, Scikit-learn, Pandas, NumPy, NLP (TF-IDF), FastAPI, Streamlit, (Optional: OpenAI / Gemini API)

---

## 📁 Project Structure
```bash id="structure_fix_01"
fake-news-detector/
│
├── app.py                 # Streamlit frontend
├── enhanced_api.py       # FastAPI backend
├── train_model.py        # Model training script
├── model.pkl             # Trained ML model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # Dependencies
└── data/                 # Dataset


---

## 🚀 How to Run

### 1. Install requirements
```bash
pip install -r requirements.txt

2.Train model
python train_model.py


3. Run backend (FastAPI)
uvicorn enhanced_api:app --reload

4. Run frontend (Streamlit)
streamlit run app.py


## 📸 Project Screenshots

### 🏠 Dashboard


### 🤖 Prediction Result
![Result](dashboard.png)

### 📝 Input & Output
![Input Output](images/input-output.png)
