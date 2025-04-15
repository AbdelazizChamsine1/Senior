
import os
import numpy as np
from PIL import Image
from pymongo import MongoClient
from bson.binary import Binary
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Constants
SAVE_PATH = "c:/Users/USER/Desktop/loading data/pictures/students"

# Initialize MTCNN and FaceNet models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["Senior_project_db"]
students_col = db["students"]
students_col.delete_many({})  # Clear existing data

# Process each student folder
for student_name in os.listdir(SAVE_PATH):
    student_path = os.path.join(SAVE_PATH, student_name)
    if not os.path.isdir(student_path) or student_name == "test":
        continue

    print(f"🧠 Processing student: {student_name}")
    student_images = []
    student_encodings = []

    for filename in os.listdir(student_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(student_path, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            image_bytes = open(img_path, "rb").read()

            # Detect and align face
            face_tensor = mtcnn(image)
            if face_tensor is None:
                print(f"❌ No face detected in {filename}")
                continue

            # Embed face
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = facenet(face_tensor)[0].cpu().numpy()

            # Store encoding
            # student_encodings.append(Binary(embedding.tobytes()))
            student_encodings.append(embedding.tolist())
            student_images.append({
                "filename": filename,
                "data": Binary(image_bytes)
            })

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            continue

    if student_images and student_encodings:
        students_col.insert_one({
            "name": student_name,
            "images": student_images,
            "face_encodings": student_encodings
        })
        print(f"✅ Stored {len(student_images)} images and {len(student_encodings)} encodings for {student_name}")
    else:
        print(f"⚠️ No valid data for {student_name}")

print("🎉 All students processed and stored in MongoDB.")
