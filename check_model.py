import os

model_path = r"C:\Users\DELL\OneDrive\Desktop\speech_emotion_model.keras"

if os.path.exists(model_path):
    print("✅ Model file exists:", model_path)
else:
    print("❌ Model file NOT found!")
