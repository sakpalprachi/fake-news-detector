import streamlit as st
import requests
import time

# ================= CONFIG =================
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide")
API_BASE_URL = "http://localhost:8000"

# ================= CSS =================
st.markdown("""
<style>
body {background-color: #f5f7fa;}

.main-title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    background: linear-gradient(90deg,#667eea,#764ba2);
    -webkit-background-clip: text;
    color: transparent;
}

.card {
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0 4px 10px rgba(0,0,0,0.1);
    margin-bottom:15px;
}

.result {
    padding:25px;
    border-radius:12px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
    color:white;
}

.fake { background:#ff4d4d; }
.real { background:#28a745; }
.uncertain { background:#ffc107; }

.footer {
    text-align:center;
    color:gray;
    margin-top:30px;
}
</style>
""", unsafe_allow_html=True)

# ================= API FUNCTIONS =================
def check_api_health():
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=5).status_code == 200
    except:
        return False


def predict_news(text, title=""):
    try:
        payload = {"text": text}
        if title:
            payload["title"] = title
        return requests.post(f"{API_BASE_URL}/predict", json=payload).json()
    except Exception as e:
        return {"error": str(e)}


def get_live_news():
    try:
        return requests.get(f"{API_BASE_URL}/live-news").json()
    except:
        return None


def fact_check_news(query):
    try:
        return requests.get(
            f"{API_BASE_URL}/fact-check",
            params={"query": query}
        ).json()
    except:
        return None


# ================= HEADER =================
st.markdown("<div class='main-title'>🔍 Fake News Detector</div>", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.title("📊 Dashboard")

    if check_api_health():
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")

    st.markdown("---")
    st.info("🤖 ML + API + Fact Check")
    
# Stop if API down
if not check_api_health():
    st.stop()

# ================= INPUT =================
st.markdown("### 📝 Analyze News")

st.markdown("<div class='card'>", unsafe_allow_html=True)

title = st.text_input("News Title (Optional)")
text = st.text_area("News Content", height=200)

st.markdown("</div>", unsafe_allow_html=True)

# ================= PREDICT =================
if st.button("🔍 Check News"):
    if not text.strip():
        st.warning("Please enter news content")
    else:
        with st.spinner("🤖 AI analyzing..."):
            time.sleep(1)
            result = predict_news(text, title)

        if "error" in result:
            st.error(result["error"])
        else:
            prediction = result["prediction"]
            confidence = result["confidence"]

            # Color logic
            if "Fake" in prediction:
                css = "fake"
                icon = "❌"
            elif "Real" in prediction:
                css = "real"
                icon = "✅"
            else:
                css = "uncertain"
                icon = "⚠️"

            # RESULT BOX
            st.markdown(f"""
            <div class='result {css}'>
            {icon} {prediction} <br>
            Confidence: {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)

            # ================= FACT CHECK =================
            st.markdown("### 🔍 Fact Check (Source Verification)")

            # 🔥 FIX: better query (title + text)
            query = (title or "") + " " + text
            fact_data = fact_check_news(query)

            if fact_data and fact_data.get("found"):
                st.success("✅ Found in trusted sources")

                for item in fact_data["results"]:
                    st.markdown(f"""
                    <div class='card'>
                    📰 <b>{item['title']}</b><br>
                    Source: {item['source']}<br>
                    <a href="{item['url']}" target="_blank">Read more</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # 🔥 FIX: better message
                st.warning("⚠️ No matching news found. Try more detailed input.")

# ================= LIVE NEWS =================
st.markdown("---")
st.header("🌍 Live News Detection")

if st.button("📰 Get Live News"):
    news_data = get_live_news()

    if not news_data or news_data.get("status") != "success":
        st.error("❌ Failed to fetch news")
    else:
        st.success(f"✅ {len(news_data['news'])} news articles loaded")

        for i, news in enumerate(news_data["news"][:5]):

            st.markdown(f"""
            <div class='card'>
            📰 <b>{news['title']}</b><br>
            <small>{news.get("description","")}</small>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"🔍 Check News {i}", key=f"btn_{i}"):

                with st.spinner("Analyzing news..."):
                    result = predict_news(
                        news["title"] + " " + str(news["description"])
                    )

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"Prediction: {result['prediction']}")
                    st.write(f"Confidence: {result['confidence']:.2f}%")

# ================= FOOTER =================
st.markdown("""
<hr>
<div class='footer'>
🚀 Fake News Detector | ML + API + Live News + Fact Check
</div>
""", unsafe_allow_html=True)