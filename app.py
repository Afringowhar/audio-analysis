import os
import streamlit as st
import tempfile
import whisper
from transformers import pipeline

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Load sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

whisper_model = load_whisper_model()
sentiment_pipeline = load_sentiment_model()

def transcribe_audio(audio_path):
    """Transcribes an audio file using Whisper."""
    result = whisper_model.transcribe(audio_path, fp16=False)
    return result["text"]

def analyze_sentiment(text):
    """Analyzes sentiment of the transcribed text."""
    sentiment = sentiment_pipeline(text)
    return sentiment[0]  # Returns a dictionary with 'label' and 'score'

# Streamlit UI
st.set_page_config(page_title="Audio Transcription & Sentiment Analysis", layout="centered")

st.title("🎤 Audio Transcription & Sentiment Analysis")
st.markdown("Upload or record an audio file and get its **transcription** along with **sentiment analysis**.")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

# Audio recorder using Streamlit's native widget
if 'audio_recorder_state' not in st.session_state:
    st.session_state.audio_recorder_state = None

record_audio = st.button("🎙️ Start Recording")

if record_audio:
    st.session_state.audio_recorder_state = st.audio_recorder()

if st.session_state.audio_recorder_state:
    st.audio(st.session_state.audio_recorder_state, format="audio/wav")
    uploaded_file = st.session_state.audio_recorder_state

if uploaded_file:
    # Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.read())
        audio_path = temp_audio.name
    
    # Transcribe and analyze sentiment
    with st.spinner("Transcribing..."):
        transcription = transcribe_audio(audio_path)

    with st.spinner("Analyzing sentiment..."):
        sentiment_result = analyze_sentiment(transcription)

    # Display results
    st.subheader("📜 Transcription:")
    st.success(transcription)

    st.subheader("📊 Sentiment Analysis:")
    st.markdown(f"**Sentiment:** {sentiment_result['label']}  \n**Confidence Score:** {sentiment_result['score']:.2f}")

    # Provide a download option for the transcription
    st.download_button("⬇️ Download Transcription", transcription, file_name="transcription.txt", mime="text/plain")

    # Cleanup temporary file
    os.remove(audio_path)
