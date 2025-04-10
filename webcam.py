import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient
from bson.binary import Binary
from sklearn.neighbors import KNeighborsClassifier

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["senior_db"]
students_col = db["students"]

# Load encodings and names
student_encodings = []
student_names = []

print("🔄 Loading face encodings from MongoDB...")
for student in students_col.find():
    if "face_encodings" in student and student["face_encodings"]:
        for encoding_data in student["face_encodings"]:
            encoding = np.frombuffer(encoding_data, dtype=np.float64)
            student_encodings.append(encoding)
            student_names.append(student["name"])

if not student_encodings:
    print("❌ No face encodings found in database! Run the encoding script first.")
    exit()

print(f"✅ Loaded {len(student_encodings)} encodings for {len(set(student_names))} students")

# Train KNN model
print("📚 Training KNN model...")
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(student_encodings, student_names)
print("✅ KNN model ready")

# Webcam setup
video_capture = cv2.VideoCapture(0)
print("📷 Press 'q' to quit...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Predict with KNN
        probabilities = knn.predict_proba([face_encoding])[0]
        best_index = np.argmax(probabilities)
        best_prob = probabilities[best_index]
        predicted_name = knn.classes_[best_index]

        # Set name and confidence
        name = predicted_name if best_prob >= 0.65 else "Unknown"
        confidence = best_prob

        # Scale up coordinates
        top, right, bottom, left = [coord * 4 for coord in face_location]

        # Draw box and label
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = f"{name} ({confidence:.2f})"
        cv2.rectangle(frame, (left, top - 25), (right, top), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the result
    cv2.imshow("🎓 Face Recognition (KNN)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
