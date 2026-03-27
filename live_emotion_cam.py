# Live emotion cam
import cv2
import numpy as np
from tensorflow.keras.models import load_model 

MODEL_PATH = 'emotion_cnn.h5'  # Path to pre-trained model

# Load trained model
print("Loading model...")
model = load_model(MODEL_PATH)

# Image size for training
IMG_SIZE = 48

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam feed

cap = cv2.VideoCapture(0)

print("Pressing 'q' will quit the program.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam not accessible")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # crop faces
        face_roi = gray[y:y+h, x:x+w]

        try:
            resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        except:
            continue # skip if too small

        # Normalize and reshape for model input
        norm = resized.astype('float32') / 255.0
        norm = np.expand_dims(norm, axis=(0, -1))  # Add batch and channel dimensions

        # prediction: probablilty of happy
        prob_sad = model.predict(norm)[0][0]

        # Confidence thresholds and labels
        neutral_low = 0.10
        neutral_high = 0.90
        if neutral_low < prob_sad < neutral_high:
            label = "Neutral"
            color = (255, 255, 255)  # White
            confidence = (1 - abs(0.5 - prob_sad) * 2) * 100

        elif prob_sad >= 0.5:
            label = "Sad"
            color = (255, 0, 0)  # Blue
            confidence = prob_sad * 100
        else:
            label = "Happy"
            color = (0, 255, 0)  # Green
            confidence = (1 - prob_sad) * 100

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({confidence:.1f}%)", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)\
                    
    cv2.imshow("Live Emotion Cam", frame)
    
    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


