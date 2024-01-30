import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Function to load audio files
def load_audio_files(path):
    audio_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    mfccs = np.mean(librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y_trimmed, sr=sr).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y_trimmed, sr=sr).T, axis=0)
    return np.hstack((mfccs, mel, chroma))

# Function to prepare the dataset
def prepare_dataset(audio_files):
    features = []
    for file in audio_files:
        try:
            feat = extract_features(file)
            features.append(feat)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    return pd.DataFrame(features)

# Function to train the model
def train_model(X, y):
    model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    model.fit(X, y)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Main script
def main():
    # Load audio files
    audio_files = load_audio_files(r'ravedss-audio\*\*.wav')

    # Prepare dataset
    df = prepare_dataset(audio_files)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
