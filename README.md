
# Speech Emotion Recognition and Music Recommendation System

This project predicts emotions from audio files and recommends a suitable song based on the detected emotion. The system is built using TensorFlow, Librosa, and Tkinter for the GUI.

├── py.py                # Main file for GUI and emotion prediction
├── train_model.py        # Script for training the model
├── label_encoder.pkl     # Encoded labels for predicting emotions
├── speech_emotion_model.keras  # Pre-trained model for prediction
├── requirements.txt      # Contains all required dependencies
└── example_audio/        # Folder with sample audio files for testing

## Prerequisites
Ensure you have Python 3.8 or above installed on your system.

## Installation
To install all required dependencies, run the following command:
```
pip install -r requirements.txt
```

### Contents of `requirements.txt`
```
tensorflow
librosa
numpy
pydub
tkinter
pickle-mixin
```


## How to Run the Project
1. **Clone the Repository**
   ```
   git clone <repository_link>
   cd <project_folder>
   ```

2. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Run the Main File**
   ```
   python py.py
   ```

4. **Select an Audio File**
   - Click the "Select Audio File" button.
   - Choose a `.wav`, `.mp3`, or `.ogg` file from the `example_audio/` folder or your system.

5. **View Results**
   - The detected emotion and a recommended song will be displayed on the screen.
     
6. **Example Audio Files**
For a better understanding of the system, try using the provided sample audio files.
  - Navigate to the example_audio/ folder.
    Select a .wav, .mp3, or .ogg file when prompted by the GUI.
    To add new audio files to your repository:
    git add example_audio/
    git commit -m "Add example audio files"
    git push origin main


## Troubleshooting
- If you encounter **"Label encoder file not found"**, ensure `label_encoder.pkl` is available in the project directory.
- If the GUI doesn’t respond, ensure Tkinter is installed correctly.

## Features
✅ Predicts six emotions: Happy, Sad, Angry, Fearful, Surprised, and Neutral.
✅ Recommends songs tailored to the detected emotion.
✅ Provides a simple and intuitive GUI for user convenience.


## Notes
- The model is trained to recognize six emotions: **Happy**, **Sad**, **Angry**, **Fearful**, **Surprised**, and **Neutral**.
- The recommended songs are predefined for each emotion.

## Future Improvements
- Adding mental health monitoring features based on emotional patterns.
- Developing personalized music therapy recommendations.
- Improving the accuracy of emotion detection.
- Enhancing the GUI for better user experience.

