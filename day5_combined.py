import cv2
import mediapipe as mp
import time
from ultralytics import YOLO

# Initialize MediaPipe
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Initialize YOLOv8
yolo_model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
prev_time = 0

face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # --- YOLOv8 Person Detection ---
    yolo_results = yolo_model(frame, classes=[0], verbose=False)
    student_count = 0
    for result in yolo_results:
        student_count = len(result.boxes)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            # Blue box = person
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                         (255, 0, 0), 2)
            cv2.putText(frame, f'{conf:.0%}',
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 0, 0), 2)

    # --- MediaPipe Face Detection ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detector.process(rgb_frame)
    face_count = 0
    if face_results.detections:
        face_count = len(face_results.detections)
        for detection in face_results.detections:
            mp_draw.draw_detection(frame, detection)

    # --- Info Panel ---
    # Black background for readability
    cv2.rectangle(frame, (5, 5), (280, 110),
                 (0, 0, 0), -1)

    cv2.putText(frame, 'CLASSROOM MONITOR v1.0',
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)

    cv2.putText(frame, f'FPS        : {int(fps)}',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 1)

    cv2.putText(frame, f'Students   : {student_count}',
                (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 0), 1)

    cv2.putText(frame, f'Faces      : {face_count}',
                (10, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 1)

    cv2.imshow('Classroom Monitoring System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

face_detector.close()
cap.release()
cv2.destroyAllWindows()