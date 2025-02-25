import streamlit as st
import whisper
import os
import tempfile
from transformers import pipeline

# Page Config
st.set_page_config(page_title="Audio Transcription & Sentiment Analysis", page_icon="🎤", layout="centered")

# Custom Styles based on Ernst & Young color palette
st.markdown("""
    <style>
        .main {
            background-color: #4169E1;
        }
        h1, h2, h3 {
            color: #333333;
            text-align: center;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            background-color: #ffe600;
            color: #333333;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #cccccc;
        }
        .transcription-box, .sentiment-box {
            background: #ffe600;
            padding: 15px;
            border-radius: 10px;
            color: #333333;
            text-align: center;
            font-size: 16px;
            font-weight: bold;
        }
        .stFileUploader label {
            color: #333333 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load Whisper model
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

# Load Sentiment Analysis Model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

sentiment_model = load_sentiment_model()

# Function to transcribe audio
def transcribe_audio(file_path):
    try:
        result = whisper_model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        return f"Error in transcription: {str(e)}"

# Function to perform sentiment analysis
def analyze_sentiment(text):
    result = sentiment_model(text)[0]
    return result["label"], result["score"]

# UI Layout
st.markdown("<h1>🎤 Audio Transcription & Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #999999;'>Upload an audio file to get its transcription and sentiment analysis.</p>", unsafe_allow_html=True)

# File Upload Section
st.markdown("### Upload an Audio File:")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "m4a"])

if uploaded_file:
    st.markdown("---")
    st.markdown("### 🎧 Audio Playback")
    
    # Save Uploaded File
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.read())
        audio_path = temp_audio.name
    
    # Audio Player
    st.audio(audio_path)
    
    # Transcription
    st.markdown("### 📝 Transcription")
    with st.spinner("Transcribing... ⏳"):
        transcription = transcribe_audio(audio_path)
    
    st.markdown(f"""
            <div class='transcription-box'>
                <p>{transcription}</p>
            </div>""", 
            unsafe_allow_html=True)
    
    # Sentiment Analysis
    if transcription.strip():
        st.markdown("### 📊 Sentiment Analysis")
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence = analyze_sentiment(transcription)
        
        st.markdown(f"""<div class='sentiment-box'>
                    <h3>Sentiment: {sentiment}</h3>
                    <p>Confidence Score: {round(confidence, 2)}</p>
                    </div>""", 
                    unsafe_allow_html=True)
    
    # Cleanup temp files
    os.unlink(audio_path)
