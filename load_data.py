import os
import cv2
import numpy as np
import face_recognition
from pymongo import MongoClient
from bson.binary import Binary
from PIL import Image
import io

# Constants
FACE_DETECTION_MODEL = "hog"  # Use the most accurate model
NUM_JITTERS = 10              # Higher jittering for better encoding accuracy
MODEL_COMPLEXITY = "large"    # Use the more complex model for higher accuracy
SAVE_PATH = "c:/Users/USER/Desktop/loading data/pictures/students"

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["senior_db"]
students_col = db["students"]
students_col.delete_many({})  # Clear existing data for fresh insert

# Helper to load image as numpy array
def load_image(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image), image_bytes

# Process each student folder
for student_name in os.listdir(SAVE_PATH):
    student_path = os.path.join(SAVE_PATH, student_name)
    if not os.path.isdir(student_path):
        continue

    print(f"🧠 Processing student: {student_name}")
    student_images = []
    student_encodings = []

    for filename in os.listdir(student_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue  # skip non-image files

        img_path = os.path.join(student_path, filename)
        image_np, image_bytes = load_image(img_path)

        # Detect face(s)
        face_locations = face_recognition.face_locations(image_np, model=FACE_DETECTION_MODEL)
        if not face_locations:
            print(f"❌ No face found in {filename}")
            continue

        # Encode face(s)
        encodings = face_recognition.face_encodings(
            image_np, face_locations, num_jitters=NUM_JITTERS, model=MODEL_COMPLEXITY
        )
        if not encodings:
            print(f"⚠️ Couldn't encode face in {filename}")
            continue

        # Save first encoding only
        encoding = encodings[0]
        student_encodings.append(Binary(encoding.tobytes()))
        student_images.append({
            "filename": filename,
            "data": Binary(image_bytes)
        })

    if student_images and student_encodings:
        # Store in MongoDB
        students_col.insert_one({
            "name": student_name,
            "images": student_images,
            "face_encodings": student_encodings
        })
        print(f"✅ Stored {len(student_images)} images and {len(student_encodings)} encodings for {student_name}")
    else:
        print(f"⚠️ No valid images/encodings for {student_name}")

print("🎉 All students processed and stored in MongoDB.")
