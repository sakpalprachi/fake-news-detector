import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import numpy as np

# ================= CONFIG =================
st.set_page_config(
    page_title="Enhanced Fake News Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# ================= CSS =================
def load_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
        }
        
        .prediction-box {
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .fake-news {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        }
        
        .real-news {
            background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(81, 207, 102, 0.3);
        }
        
        .uncertain-news {
            background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(255, 167, 38, 0.3);
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            margin: 0.5rem 0;
        }
        
        .model-comparison {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .ai-explanation {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            font-weight: bold;
            border-radius: 25px;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .confidence-gauge {
            height: 200px;
        }
    </style>
    """, unsafe_allow_html=True)

# ================= API FUNCTIONS =================
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_news(text, title="", use_ensemble=True):
    try:
        payload = {"text": text}
        if title:
            payload["title"] = title
        
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def get_model_comparison():
    try:
        response = requests.get(f"{API_BASE_URL}/model/comparison", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def get_ai_analysis(text, title=""):
    try:
        payload = {"text": text}
        if title:
            payload["title"] = title
        
        response = requests.post(f"{API_BASE_URL}/ai/analyze", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def summarize_news(text, title=""):
    try:
        payload = {"text": text}
        if title:
            payload["title"] = title
        
        response = requests.post(f"{API_BASE_URL}/ai/summarize", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def get_live_news():
    try:
        response = requests.get(f"{API_BASE_URL}/live-news", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

# ================= VISUALIZATION =================
def create_confidence_gauge(confidence, prediction):
    colors = {
        "Fake News ❌": "#ff6b6b",
        "Real News ✅": "#51cf66", 
        "Uncertain ⚠️": "#ffa726"
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confidence Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': colors.get(prediction, "#667eea")},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': colors.get(prediction, "#667eea")}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"},
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0}
    )
    
    return fig

def create_model_comparison_chart(model_data):
    if model_data.get("status") != "success":
        return None
    
    models = model_data.get("models", {})
    model_names = list(models.keys())
    accuracies = [models[name] * 100 for name in model_names]
    
    colors = px.colors.qualitative.Set3
    fig = px.bar(
        x=model_names,
        y=accuracies,
        color=colors[:len(model_names)],
        labels={
            'x': 'Model',
            'y': 'Accuracy (%)'
        }
    )
    
    fig.update_layout(
        title="Model Performance Comparison",
        yaxis=dict(range=[0, 100]),
        height=400
    )
    
    return fig

# ================= MAIN APP =================
def main():
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Enhanced Fake News Detector</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛 Control Panel")
        
        # API Status
        if check_api_health():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.warning("Please start enhanced_api.py")
        
        st.markdown("---")
        
        # Model Selection
        st.header("🤖 Model Selection")
        use_ensemble = st.radio(
            "Choose Prediction Method",
            ["Ensemble (Recommended)", "Best Single Model"],
            help="Ensemble combines multiple models for better accuracy"
        )
        
        # Model Comparison
        if st.button("📊 View Model Comparison"):
            model_data = get_model_comparison()
            if model_data.get("status") == "success":
                st.subheader("📈 Model Performance")
                
                # Display comparison chart
                fig = create_model_comparison_chart(model_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display model details
                st.markdown("### 🏆 Model Rankings")
                for model_name, accuracy in model_data.get("models", {}).items():
                    st.metric(
                        model_name.replace('_', ' ').title(),
                        f"{accuracy:.2f}%"
                    )
                
                # Best model info
                best_model = model_data.get("best_model", "Unknown")
                st.info(f"🏆 Best Model: {best_model.replace('_', ' ').title()}")
            else:
                st.error(model_data.get("error", "Failed to get model comparison"))
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 News Analysis")
        
        # Input section
        with st.expander("📄 Input News Content", expanded=True):
            title = st.text_input(
                "News Title (Optional)",
                placeholder="Enter the news headline here...",
                help="Adding a title can improve prediction accuracy"
            )
            
            text = st.text_area(
                "News Content *",
                placeholder="Paste or type the full news article content here...",
                height=150,
                help="The more detailed the content, the better the prediction"
            )
        
        # Prediction buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🔍 Analyze News", type="primary", use_container_width=True):
                if not text.strip():
                    st.error("❌ Please enter news content")
                else:
                    with st.spinner("🤖 Analyzing with Enhanced AI..."):
                        result = predict_news(text, title, use_ensemble=(use_ensemble == "Ensemble (Recommended)"))
                        
                        if "error" in result:
                            st.error(f"❌ Error: {result['error']}")
                        else:
                            st.session_state.prediction_result = result
        
        with col_btn2:
            if st.button("🧠 AI Explanation", use_container_width=True):
                if not text.strip():
                    st.error("❌ Please enter news content first")
                else:
                    with st.spinner("🤖 AI Analysis in Progress..."):
                        ai_result = get_ai_analysis(text, title)
                        
                        if "error" in ai_result:
                            st.error(f"❌ AI Error: {ai_result['error']}")
                        else:
                            st.session_state.ai_result = ai_result
        
        with col_btn3:
            if st.button("📝 Summarize", use_container_width=True):
                if not text.strip():
                    st.error("❌ Please enter news content first")
                else:
                    with st.spinner("🤖 Summarizing with AI..."):
                        summary_result = summarize_news(text, title)
                        
                        if "error" in summary_result:
                            st.error(f"❌ Summary Error: {summary_result['error']}")
                        else:
                            st.session_state.summary_result = summary_result
    
    with col2:
        st.header("📊 Results & Insights")
        
        # Display prediction results
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            
            # Prediction box
            prediction = result.get('prediction', 'Unknown')
            confidence = result.get('confidence', 0)
            probabilities = result.get('probabilities', {})
            method_used = result.get('method_used', 'Unknown')
            suspicious_words = result.get('suspicious_words', [])
            
            # Determine CSS class
            if "Fake" in prediction:
                css_class = "fake-news"
            elif "Real" in prediction:
                css_class = "real-news"
            else:
                css_class = "uncertain-news"
            
            st.markdown(f"""
            <div class='prediction-box {css_class}'>
                {prediction}
                <br>
                <small>Confidence: {confidence:.2f}% | Method: {method_used}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence gauge
            fig = create_confidence_gauge(confidence, prediction)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics
            st.markdown("### 📈 Detailed Analysis")
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.metric("Prediction", prediction)
            
            with col_metric2:
                st.metric("Confidence", f"{confidence:.2f}%")
            
            with col_metric3:
                st.metric("Method", method_used)
            
            # Probabilities
            st.markdown("### 🎯 Probability Distribution")
            prob_df = pd.DataFrame([
                {"Category": "Fake News", "Probability": f"{probabilities.get('fake', 0):.2f}%"},
                {"Category": "Real News", "Probability": f"{probabilities.get('real', 0):.2f}%"}
            ])
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            # Suspicious words
            if suspicious_words:
                st.markdown("### ⚠️ Suspicious Keywords Detected")
                for word in suspicious_words:
                    st.warning(f"🔍 {word}")
        
        # Display AI analysis
        if 'ai_result' in st.session_state:
            ai_result = st.session_state.ai_result
            
            st.markdown("### 🤖 AI-Powered Analysis")
            st.markdown(f"""
            <div class='ai-explanation'>
                <strong>🧠 AI Explanation:</strong><br>
                {ai_result.get('ai_explanation', 'AI analysis not available')}
            </div>
            """, unsafe_allow_html=True)
            
            # Fact check suggestions
            if ai_result.get('fact_check_suggestions'):
                st.markdown("### 🔍 Fact-Check Suggestions")
                for suggestion in ai_result.get('fact_check_suggestions', []):
                    st.info(f"💡 {suggestion}")
        
        # Display summary
        if 'summary_result' in st.session_state:
            summary_result = st.session_state.summary_result
            
            st.markdown("### 📝 AI Summary")
            st.success(f"""
            **Summary:** {summary_result.get('summary', 'Summary not available')}
            <br><small>Word Count: {summary_result.get('word_count', 0)} words</small>
            """)
            
            if summary_result.get('ml_prediction'):
                st.info(f"🤖 ML Classification: {summary_result.get('ml_prediction')}")
    
    # Live News Section
    st.markdown("---")
    st.header("🌍 Live News Analysis")
    
    if st.button("📰 Get Live News"):
        with st.spinner("🔄 Fetching live news..."):
            news_data = get_live_news()
            
            if news_data.get("status") == "success":
                st.success(f"✅ {news_data.get('total', 0)} news articles loaded")
                
                # Display news articles with analysis
                for i, news in enumerate(news_data.get("news", [])[:5]):
                    with st.expander(f"📰 {news.get('title', 'Untitled')[:50]}...", expanded=False):
                        st.write(news.get('description', 'No description available'))
                        
                        # Quick analysis from live news
                        if news.get('quick_analysis'):
                            analysis = news['quick_analysis']
                            st.markdown(f"""
                            <div class='model-comparison'>
                                <strong>🤖 Quick Analysis:</strong> {analysis.get('prediction', 'Unknown')}<br>
                                <strong>📊 Confidence:</strong> {analysis.get('confidence', 0):.2f}%<br>
                                <strong>🔍 Suspicious Words:</strong> {', '.join(analysis.get('suspicious_words', []))}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Analyze button
                        if st.button(f"🔍 Full Analysis {i}", key=f"live_{i}"):
                            full_text = f"{news.get('title', '')} {news.get('description', '')}"
                            with st.spinner("🤖 Analyzing with Enhanced AI..."):
                                result = predict_news(full_text, use_ensemble=True)
                                
                                if "error" in result:
                                    st.error(f"❌ Analysis Error: {result['error']}")
                                else:
                                    st.session_state.live_analysis_result = result
            else:
                st.error(f"❌ Failed to fetch news: {news_data.get('error', 'Unknown error')}")
    
    # Display live analysis results
    if 'live_analysis_result' in st.session_state:
        result = st.session_state.live_analysis_result
        st.markdown("### 📊 Live News Analysis Results")
        
        prediction = result.get('prediction', 'Unknown')
        confidence = result.get('confidence', 0)
        method_used = result.get('method_used', 'Unknown')
        
        if "Fake" in prediction:
            css_class = "fake-news"
        elif "Real" in prediction:
            css_class = "real-news"
        else:
            css_class = "uncertain-news"
        
        st.markdown(f"""
        <div class='prediction-box {css_class}'>
            {prediction}
            <br>
            <small>Confidence: {confidence:.2f}% | Method: {method_used}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <strong>🚀 Enhanced Fake News Detector v2.0</strong><br>
        <small>Multiple ML Algorithms • Ensemble Method • AI Integration • Real-time Analysis</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
