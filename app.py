
from flask import Flask, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import datetime
import numpy as np

# Setup
app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=30, keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["Senior_project_db"]
quotes_collection = db["quotes"]
students_collection = db["students"]

# === Helper: Get random quote ===
def get_random_quote():
    quote_doc = quotes_collection.aggregate([{"$sample": {"size": 1}}]).next()
    return quote_doc["text"]

# === Face Recognition ===
def recognize_face():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, None

    # Convert image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img_rgb)
    if face_tensor is None:
        return None, None

    # Get embedding
    embedding = facenet(face_tensor.unsqueeze(0).to(device)).detach().cpu().numpy()[0]

    known_students = students_collection.find()
    min_dist = float('inf')
    identity = "Unknown"

    for student in known_students:
        if 'face_encodings' not in student:
            continue

        for db_embedding_list in student['face_encodings']:
            db_embedding = np.array(db_embedding_list)
            dist = np.linalg.norm(embedding - db_embedding)
            if dist < min_dist and dist < 0.9:
                min_dist = dist
                identity = student['name']

    if identity != "Unknown":
        return identity, get_random_quote()
    return None, None

# === Route ===
@app.route("/recognize", methods=["GET"])
def recognize():
    name, quote = recognize_face()
    if name:
        return jsonify({
            "status": "success",
            "name": name,
            "quote": quote
        })
    else:
        return jsonify({
            "status": "unknown",
            "message": "Face not recognized"
        })

if __name__ == "__main__":
    app.run(debug=True)
