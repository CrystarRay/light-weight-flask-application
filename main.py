from flask import Flask, request, jsonify
import requests
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mutils import convert
import os

# Load the model from local directory
MODEL_NAME = "trained_with_20_files"
model = tf.keras.models.load_model(f"./model/{MODEL_NAME}/model.h5")

# Define MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Flask app
app = Flask(__name__)
app.debug = True

@app.route('/', methods=['POST'])
def predict():
    url = request.get_json()['url']
    filename = download_file(url)

    try:
        # Process video
        mae_losses = process_video(filename)
        # Delete the file after processing
        os.remove(filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Return the result
    if mae_losses:
        average_loss = sum(mae_losses) / len(mae_losses)
        result = {"average_loss": average_loss, "all_losses": mae_losses}
    else:
        result = "No pose detected"

    return jsonify(result)

def download_file(url):
    r = requests.get(url, stream=True)
    r.raise_for_status()  # Raise exception if invalid response
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")  # Create a temp file with .mp4 extension
    with open(temp.name, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return temp.name

def process_video(video_file):
    cap = cv2.VideoCapture(video_file)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mae_losses = []  # Store all the losses
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            if results.pose_landmarks is not None:
                converted_landmarks = convert(results.pose_world_landmarks.landmark)
                mae_loss = model.evaluate(np.array([converted_landmarks]), np.array([converted_landmarks]), verbose=0)[0]
                mae_losses.append(mae_loss)
    finally:
        cap.release()
    return mae_losses

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
