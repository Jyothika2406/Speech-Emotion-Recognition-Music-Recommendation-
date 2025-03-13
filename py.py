import os 
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import tensorflow as tf
import pickle
from pydub import AudioSegment

# Define paths
model_path = r"C:\Users\DELL\OneDrive\Desktop\speech_emotion_model.keras"  
label_encoder_path = r"C:\Users\DELL\OneDrive\Desktop\speech emotion music recomandation\label_encoder.pkl"

# Load trained model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = tf.keras.models.load_model(model_path)

# Load Label Encoder
if not os.path.exists(label_encoder_path):
    raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# Emotion-to-Music Mapping
emotion_music = {
    "Happy": "Pharrell Williams - Happy 🎶",
    "Sad": "Adele - Someone Like You 😢",
    "Angry": "Eminem - Lose Yourself 🔥",
    "Fearful": "Billie Eilish - Everything I Wanted 😨",
    "Surprised": "Coldplay - Viva La Vida 🎵",
    "Neutral": "John Legend - All of Me 🎼",
}

# Function to extract MFCC features with correct shape
def extract_features(file_path, max_pad_length=100):
    try:
        audio, sample_rate = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Normalize MFCC features
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc) 

        # Ensure MFCC shape is (40, max_pad_length)
        if mfcc.shape[1] < max_pad_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_length - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_length]

        return np.expand_dims(mfcc, axis=0)  # Shape will be (1, 40, 100)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


# Function to predict emotion and recommend a song
def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        return "Error: Could not extract features", "No song available"

    features = np.expand_dims(features, axis=-1)  # Ensuring shape is (1, 40, 100, 1) if required

    predictions = model.predict(features)
    predicted_class = np.argmax(predictions)
    predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
    recommended_song = emotion_music.get(predicted_emotion, "No song available")

    return predicted_emotion, recommended_song


# GUI Application
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3;*.ogg")])
    if not file_path:
        return
    
    # Convert to WAV if needed
    if not file_path.endswith(".wav"):
        audio = AudioSegment.from_file(file_path)
        file_path = file_path + ".wav"
        audio.export(file_path, format="wav")
    
    emotion, song = predict_emotion(file_path)
    label_result.config(text=f"Predicted Emotion: {emotion}\nRecommended Song: {song}")

# Create Tkinter window
root = tk.Tk()
root.title("Speech Emotion Recognition & Music Recommendation")
root.geometry("500x300")

btn_select = tk.Button(root, text="Select Audio File", command=open_file)
btn_select.pack(pady=20)

label_result = tk.Label(root, text="", font=("Arial", 12))
label_result.pack(pady=20)

root.mainloop()
