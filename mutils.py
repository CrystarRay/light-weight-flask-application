landmark_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]

# convert landmarks to only selected landmarks
def convert(landmarks):
    result = []
    for index in landmark_indices:
        landmark = landmarks[index]
        result.extend([landmark.x, landmark.y, landmark.z])
    return result