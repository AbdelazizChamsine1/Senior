
import cv2
import numpy as np
from PIL import Image
import csv
import datetime
import os
from pymongo import MongoClient
from bson.binary import Binary
from sklearn.neighbors import KNeighborsClassifier
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Constants
CONFIDENCE_THRESHOLD = 0.99
MARGIN = 30
TODAY = datetime.datetime.now().strftime("%Y-%m-%d")
ATTENDANCE_FILE = f"attendance_{TODAY}.csv"

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=MARGIN, keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["Senior_project_db"]
students_col = db["students"]

# Load all individual face encodings
student_encodings = []
student_names = []

print("🔄 Loading all face encodings from MongoDB...")
for student in students_col.find():
    if "face_encodings" in student:
        for enc in student["face_encodings"]:
            encoding = np.array(enc, dtype=np.float32)
            student_encodings.append(encoding)
            student_names.append(student["name"])

if not student_encodings:
    print("❌ No face encodings found.")
    exit()

# Train KNN
print("🧠 Training KNN model...")
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(student_encodings, student_names)
print("✅ KNN model ready")

# Setup attendance CSV
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Time", "Student", "Confidence"])

marked_students = set()

# Start webcam
video_capture = cv2.VideoCapture(0)
print("📷 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    boxes, _ = mtcnn.detect(pil_img)
    faces = mtcnn(pil_img)

    if boxes is not None and faces is not None:
        for i, face_tensor in enumerate(faces):
            if face_tensor is None:
                continue

            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = facenet(face_tensor).cpu().numpy()[0]

            distances, indices = knn.kneighbors([embedding], n_neighbors=3)
            mean_distance = np.mean(distances[0])

            probabilities = knn.predict_proba([embedding])[0]
            best_index = np.argmax(probabilities)
            best_prob = probabilities[best_index]
            predicted_name = knn.classes_[best_index]

            # Set a new distance threshold
            DISTANCE_THRESHOLD = 0.9

            if best_prob >= CONFIDENCE_THRESHOLD and mean_distance < DISTANCE_THRESHOLD:
                label = f"{predicted_name} ({best_prob:.2f})"
                color = (0, 255, 0)
                
                if predicted_name not in marked_students:
                    marked_students.add(predicted_name)
                    now = datetime.datetime.now()
                    with open(ATTENDANCE_FILE, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            now.strftime("%Y-%m-%d"),
                            now.strftime("%H:%M:%S"),
                            predicted_name,
                            f"{best_prob:.4f}"
                        ])
            else:
                predicted_name = "Unknown"
                label = f"{predicted_name} ({best_prob:.2f}, dist {mean_distance:.2f})"
                color = (0, 0, 255)

            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, cv2.FILLED)
            cv2.putText(frame, label, (x1 + 6, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("🎓 Face Recognition + Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
