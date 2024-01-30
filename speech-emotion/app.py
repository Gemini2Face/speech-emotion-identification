import streamlit as st
from audiorecorder import audiorecorder
import librosa
import numpy as np
from joblib import load

# Load your pre-trained model
model = load("speech-emotion/model/emotion_classifier_model.joblib")


# Function to preprocess the audio file
def preprocess_audio(file):
    # Load and preprocess your audio file as per your model's requirement
    y, sr = librosa.load(file, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    stft = np.abs(librosa.stft(y_trimmed))
    mfccs = np.mean(librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y_trimmed, sr=sr).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    return np.hstack((mfccs, mel, chroma))


# Function to make prediction
def predict(audio_data):
    features = preprocess_audio(audio_data)
    # Predict and return the emotion
    prediction = model.predict([features])
    return prediction


# Streamlit UI components
st.title("Audio Emotion Recognition")

# Audio recorder component
st.write("Record your voice")
audio = audiorecorder("Click to record", "Click to stop recording")

if audio is not None and len(audio) > 0:
    # Play the recorded audio
    st.audio(audio.export().read(), format="audio/wav")

    # Save the audio to a file
    audio.export("recorded_audio.wav", format="wav")

    # Predict emotion from the recorded audio
    prediction = predict("recorded_audio.wav")
    st.write(f"Predicted Emotion: {prediction}")

    # Optional: Display audio properties
    st.write(
        f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds"
    )
