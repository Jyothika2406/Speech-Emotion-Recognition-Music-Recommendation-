import pickle
from sklearn.preprocessing import LabelEncoder

# Define the emotions used during training
emotions = ["Happy", "Sad", "Angry", "Fearful", "Surprised", "Neutral"]

# Create and fit the LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(emotions)

# Save it as a pickle file
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Label encoder saved successfully!")

label_encoder_path = r"C:\Users\DELL\OneDrive\Desktop\label_encoder.pkl"


