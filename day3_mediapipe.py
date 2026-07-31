import cv2
import mediapipe as mp
import time

# Initialize MediaPipe
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev_time = 0

with mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5) as face_detector:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # MediaPipe needs RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        
        # Detect and draw faces
        face_count = 0
        if results.detections:
            face_count = len(results.detections)
            for detection in results.detections:
                mp_draw.draw_detection(frame, detection)
        
        # Display info
        cv2.putText(frame, f'FPS: {int(fps)}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
                    
        cv2.putText(frame, f'Faces Detected: {face_count}',
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        
        cv2.imshow('Classroom Monitor - MediaPipe', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

