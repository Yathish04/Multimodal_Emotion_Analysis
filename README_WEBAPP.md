# Multimodal Emotion Detection Web App

This is a Flask-based web application for Multimodal Emotion Detection (Text, Voice, Facial, and Fusion).

## Features
- **Text Emotion**: Enter text to analyze emotion using DistilRoberta.
- **Voice Emotion**: Record audio in the browser or upload a WAV file.
- **Facial Emotion**: Real-time facial emotion detection via webcam.
- **Fusion Module**: Combines all three modalities for a robust emotion prediction.
- **Logs**: View history of emotion detections.

## Setup

1.  **Install Dependencies**:
    Ensure you have the requirements installed.
    ```bash
    pip install -r requirements.txt
    pip install flask flask-cors
    ```
    *Note: You might need `tf-keras` if using newer TensorFlow versions.*

2.  **Run the Application**:
    ```bash
    python app.py
    ```

3.  **Access the App**:
    Open your browser and go to `http://localhost:5000`.

## Usage
- **Dashboard**: Shows the Fusion module output. It uses your webcam and microphone (server-side).
- **Text**: Type text and click Analyze.
- **Voice**: Click the microphone icon to record (requires browser permission) or upload a file.
- **Face**: Shows the webcam feed with facial emotion detection only.
- **Logs**: Check past predictions.

## Security & Data
- Data is stored in a local SQLite database (`emotion_data.db`).
- Inputs are validated before processing.
