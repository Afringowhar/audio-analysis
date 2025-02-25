    # import streamlit as st
    # import whisper
    # import torch
    # import os
    # import tempfile
    # from transformers import pipeline
    # import base64

    # # Page Config
    # st.set_page_config(page_title="Audio Transcription & Sentiment Analysis", page_icon="🎤", layout="wide")

    # # Load Whisper model
    # @st.cache_resource
    # def load_whisper():
    #     return whisper.load_model("base")

    # whisper_model = load_whisper()

    # # Load Sentiment Analysis Model
    # @st.cache_resource
    # def load_sentiment_model():
    #     return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    # sentiment_model = load_sentiment_model()

    # # Function to transcribe audio
    # def transcribe_audio(file_path):
    #     try:
    #         result = whisper_model.transcribe(file_path)
    #         return result["text"]
    #     except Exception as e:
    #         return f"Error in transcription: {str(e)}"

    # # Function to perform sentiment analysis
    # def analyze_sentiment(text):
    #     result = sentiment_model(text)[0]
    #     return result["label"], result["score"]

    # # Function to show audio player
    # def show_audio_player(file_path):
    #     with open(file_path, "rb") as audio_file:
    #         audio_bytes = audio_file.read()
    #     st.audio(audio_bytes)

    # # UI Layout
    # st.markdown("<h1 style='text-align: center;'>🎤 Audio Transcription & Sentiment Analysis</h1>", unsafe_allow_html=True)
    # st.markdown("<p style='text-align: center;'>Upload an audio file and get its transcription and sentiment analysis.</p>", unsafe_allow_html=True)

    # # Upload Audio File
    # st.markdown("### Upload an Audio File:")
    # uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "m4a"])

    # if uploaded_file:
    #     st.markdown("### 🎧 Audio Playback")
        
    #     # Save Uploaded File
    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
    #         temp_audio.write(uploaded_file.read())
    #         audio_path = temp_audio.name

    #     show_audio_player(audio_path)

    #     # Transcription
    #     st.markdown("### 📝 Transcription")
    #     with st.spinner("Transcribing... ⏳"):
    #         transcription = transcribe_audio(audio_path)
        
    #     st.write(transcription)

    #     # Sentiment Analysis
    #     if transcription.strip():
    #         st.markdown("### 📊 Sentiment Analysis")
    #         with st.spinner("Analyzing sentiment..."):
    #             sentiment, confidence = analyze_sentiment(transcription)

    #         st.write(f"**Sentiment:** {sentiment}")
    #         st.write(f"**Confidence Score:** {round(confidence, 2)}")

    #     # Cleanup temp files
    #     os.unlink(audio_path)



import streamlit as st
import whisper
import torch
import os
import tempfile
from transformers import pipeline
import base64

# Page Config
st.set_page_config(page_title="Audio Transcription & Sentiment Analysis", page_icon="🎤", layout="centered")

# Custom Styles
st.markdown("""
    <style>
        .main {
            background-color: #001A6E;
        }
        h1, h2, h3 {
            color: #2c3e50;
            text-align: center;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            background-color: #3498db;
            color: white;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #2980b9;
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
st.markdown("<p style='text-align: center;'>Upload an audio file to get its transcription and sentiment analysis.</p>", unsafe_allow_html=True)

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
            <div style='background: #FFB200; padding: 10px; border-radius: 10px;'>
                <p style='font-size:16px; text-align:center;'>{transcription}</p></div>""", 
                unsafe_allow_html=True)
    
    # Sentiment Analysis
    if transcription.strip():
        st.markdown("### 📊 Sentiment Analysis")
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence = analyze_sentiment(transcription)
        
        st.markdown(f"""<div style='background: #FFB200; padding: 10px; border-radius: 10px;'>
                    <h3 style='text-align:center;'>Sentiment: {sentiment}</h3>
                    <p style='text-align:center; font-size:16px;'>Confidence Score: {round(confidence, 2)}</p>
                    </div>""", 
                    unsafe_allow_html=True)
    
    # Cleanup temp files
    os.unlink(audio_path)
