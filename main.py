from flask import Flask, request, jsonify
import requests
import tempfile
import time
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mutils import convert
import stats_model_testing as stats
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
    start_time = time.time()
    print("start time:0.0 seconds")

    url = request.get_json()['url']
    filename = download_file(url)

    download_time = time.time()
    download_time = download_time - start_time
    print(f"finished download video: {download_time:.4f} seconds")

    try:
        # Process video
        mae_losses = process_video(filename, start_time)
        os.remove(filename)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


    if mae_losses:
        result = stats.classify_losses(mae_losses, "./threshold_m1.txt")
    else:
        result = "No pose detected"

    end_time = time.time()
    finish_everything = end_time - start_time
    print(f"finished everything: {finish_everything:.4f} seconds")

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

def process_video(video_file, start_time):
    cap = cv2.VideoCapture(video_file)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mae_losses = []  # Store all the losses
    try:
        frame_data = []
        count1 = 0
        count2 = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            if results.pose_landmarks is not None:
                converted_landmarks = convert(results.pose_world_landmarks.landmark)
                frame_data.append(converted_landmarks)
                
                
                key_point_coordintes = time.time()
                key_point_coordintes = key_point_coordintes - start_time
                count1 += 1
                print(f" finish key point coordintes {count1}: {key_point_coordintes:.4f} seconds")
                
                if len(frame_data) == 16:
                    mae_loss = model.evaluate(np.array([frame_data]), np.array([frame_data]), verbose=0)[0]
                    mae_losses.append(mae_loss)
                    frame_data = []
                    
                    finish_stability_model = time.time()
                    finish_stability_model = finish_stability_model - start_time
                    count2 += 1
                    print(f"finish stability model {count2}: {finish_stability_model:.4f} seconds")
                    
    finally:
        cap.release()
    return mae_losses

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
