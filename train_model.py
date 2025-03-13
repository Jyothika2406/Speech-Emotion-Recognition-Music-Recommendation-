import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import numpy as np

# Generate dummy data
X_train = np.random.rand(100, 40, 100)  # 100 samples, 40 MFCC features, 100 time steps
y_train = np.random.randint(0, 6, 100)  # 100 labels (6 emotion classes)

# Create a simple model
model = Sequential([
    Flatten(input_shape=(40, 100)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')  # 6 emotion categories
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5)

# Save the trained model
model.save(r"C:\Users\DELL\OneDrive\Desktop\speech_emotion_model.keras")

print("✅ Model trained and saved successfully!")
