import cv2
from deepface import DeepFace
import numpy as np
import pandas as pd

def analyze_and_display(frame, prev_emotion):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            try:
                results = DeepFace.analyze(face, actions=['emotion'])
                if isinstance(results, list):
                    result = results[0]
                else:
                    result = results

                if 'dominant_emotion' in result:
                    dominant_emotion = result['dominant_emotion']
                    if dominant_emotion != prev_emotion:
                        print(f"Emotion: {dominant_emotion}")
                        prev_emotion = dominant_emotion
                    cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    print("Error: 'dominant_emotion' not found in result:", result)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            except Exception as e:
                print("Error in DeepFace analysis:", e)
    else:
        if prev_emotion is not None:
            prev_emotion = None

    return frame, prev_emotion

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

prev_emotion = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame, prev_emotion = analyze_and_display(frame, prev_emotion)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
