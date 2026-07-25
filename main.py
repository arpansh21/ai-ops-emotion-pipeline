import tensorflow as tf
import numpy as np
import cv2
import json

print("🚀 Running NEW webcam code...")

# Load model
model = tf.keras.models.load_model("model.h5")

# Load labels
with open("labels.json", "r") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def preprocess_face(face):
    if face.size == 0:
        return None

    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        processed = preprocess_face(face)
        if processed is None:
            continue

        prediction = model.predict(processed, verbose=0)

        label = labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        text = f"{label} ({confidence:.2f})"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()