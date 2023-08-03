import tempfile
import time

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from firebase_admin import credentials, initialize_app, storage
from werkzeug.exceptions import abort
import requests
from mutils import convert
import os
from google.cloud import storage as gcs

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mobility-scooter-app-firebase-adminsdk-twji0-182bd76cd8.json'
os.environ['BUCKET_NAME'] = 'trainingvideo-project-123-20230726'

MODEL_NAME = "trained_with_20_files"
TIMESTAMPS = 16
bucket_name = os.getenv('BUCKET_NAME')

# Load your model
storage_client = gcs.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(f"model/{MODEL_NAME}/model.h5")
blob.download_to_filename("/tmp/model.h5")
model = tf.keras.models.load_model("/tmp/model.h5")

# Download Firebase credentials to local environment
firebase_blob = bucket.blob('mobility-scooter-app-firebase-adminsdk-twji0-182bd76cd8.json')
firebase_blob.download_to_filename('/tmp/mobility-scooter-app-firebase-adminsdk-twji0-182bd76cd8.json')

# Initialize Firebase
cred = credentials.Certificate('/tmp/mobility-scooter-app-firebase-adminsdk-twji0-182bd76cd8.json')
default_app = initialize_app(cred, {'storageBucket': bucket_name})
bucket = storage.bucket()


def download_file(url, filename):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return filename

def download_file(url, filename):
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)

def process_video(video_file):
    cap = cv2.VideoCapture(video_file)
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
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

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    url = data.get('url')

    video_file = download_file(url, "temp_video_file")

    try:
        mae_losses = process_video(video_file)
    except Exception as e:
        abort(500, description="Failed to process video: " + str(e))

    # Clean up temporary file
    time.sleep(5)  # add delay
    os.remove(video_file)

    return jsonify({'mae_losses': mae_losses})

if __name__ == '__main__':
    app.run(debug=True)
