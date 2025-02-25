import asyncio
import torch
import streamlit as st
from transformers import pipeline
import speech_recognition as sr
from pydub import AudioSegment

# Ensure an event loop is running
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# Streamlit page configuration (must be first Streamlit command)
st.set_page_config(page_title="Audio Transcription & Sentiment Analysis", layout="centered")

# Title and description
st.title("🎤 Audio Transcription & Sentiment Analysis")
st.markdown("Upload or record an audio file and get its **transcription and sentiment analysis**.")

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to transcribe audio
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(audio_path)
    audio.export("temp.wav", format="wav")  # Convert to WAV format

    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError:
            return "Error with the speech recognition service"

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = sentiment_pipeline(text)
    return sentiment[0]  # Returns a dictionary with 'label' and 'score'

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save uploaded file
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Transcribe audio
    transcription = transcribe_audio(file_path)
    st.subheader("Transcription:")
    st.write(transcription)

    # Sentiment analysis
    if transcription and transcription != "Could not understand the audio":
        sentiment_result = analyze_sentiment(transcription)
        st.subheader("Sentiment Analysis:")
        st.write(f"**Sentiment:** {sentiment_result['label']}")
        st.write(f"**Confidence Score:** {sentiment_result['score']:.2f}")
