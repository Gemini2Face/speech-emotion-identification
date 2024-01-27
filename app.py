import streamlit as st
import numpy as np
import librosa
import io
from tensorflow.keras.models import load_model
import joblib

# Load the model and the label encoder
model = load_model("C://Users//Malhan//Downloads//Test Audios//emotion_recognition_model.h5")
label_encoder = joblib.load("C://Users//Malhan//Downloads//Test Audios//label_encoder.pkl")


# Function to process audio for prediction
def process_audio_for_prediction(audio_data, target_length=227):
    y, sr = librosa.load(audio_data, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=50)
    S = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, n_mels=19)
    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
    if S_db_mel.shape[1] < target_length:
        pad_width = target_length - S_db_mel.shape[1]
        S_db_mel_padded = np.pad(S_db_mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        S_db_mel_padded = S_db_mel[:, :target_length]
    return S_db_mel_padded.mean(axis=1)

# Function to predict emotion from features
def predict_emotion(features, model, label_encoder):
    features_scaled = features.reshape(1, 1, -1)
    prediction = model.predict(features_scaled)
    predicted_label = np.argmax(prediction, axis=1)[0]
    predicted_emotion = label_encoder.inverse_transform([predicted_label])
    return predicted_emotion[0]

# Streamlit app
st.title("Emotion Recognition from Voice")

# File uploader for audio
uploaded_file = st.file_uploader("Upload an audio file for emotion detection", type=["wav", "mp3"])

# Process and predict emotion
if st.button('Find Emotion') and uploaded_file is not None:
    audio_data = io.BytesIO(uploaded_file.read())
    features = process_audio_for_prediction(audio_data)
    predicted_emotion = predict_emotion(features, model, label_encoder)
    st.write(f"Predicted Emotion: {predicted_emotion}")