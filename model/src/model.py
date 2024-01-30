import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import pickle

# Set visual theme for plots
sns.set_theme(style="white")

# Define emotion codes
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def load_audio_files(path_pattern):
    """Load audio files from a given path pattern."""
    return glob(path_pattern)


def extract_feature(file_name):
    """Extract features from a single file."""
    y, sr = librosa.load(file_name, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    stft = np.abs(librosa.stft(y_trimmed))
    mfccs = np.mean(librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y_trimmed, sr=sr).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    return np.hstack((mfccs, mel, chroma))


def prepare_dataset(file_paths):
    """Prepare the dataset from audio files."""
    data = []
    for file in file_paths:
        features = extract_feature(file)
        emotion = EMOTIONS[file.split("-")[2]]
        data.append([features, emotion])
    return pd.DataFrame(data, columns=["Features", "Emotion"])


# Load and prepare the dataset
audio_files = load_audio_files("dataset/**/*.wav")
dataset = prepare_dataset(audio_files)

# Split the dataset into features and labels
X = np.array(dataset["Features"].tolist())
y = np.array(dataset["Emotion"].tolist())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
model = MLPClassifier(
    alpha=0.01,
    batch_size=256,
    epsilon=1e-08,
    hidden_layer_sizes=(300,),
    learning_rate="adaptive",
    max_iter=500,
)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Saving the model using joblib
dump(model, "emotion_classifier_model.joblib")

# Pickle
with open("emotion_classifier_model.pkl", "wb") as file:
    pickle.dump(model, file)
