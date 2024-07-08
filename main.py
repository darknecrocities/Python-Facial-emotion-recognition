import cv2
from deepface import DeepFace

def analyze_and_display(frame, prev_emotion):
    try:
        results = DeepFace.analyze(frame, actions=['emotion'])

        if isinstance(results, list):
            result = results[0]
        else:
            result = results
        
        if 'dominant_emotion' in result:
            dominant_emotion = result['dominant_emotion']
            if dominant_emotion != prev_emotion:
                print(f"Emotion: {dominant_emotion}")
                prev_emotion = dominant_emotion
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            print("Error: 'dominant_emotion' not found in result:", result)
    except Exception as e:
        print("Error in DeepFace analysis:", e)
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
