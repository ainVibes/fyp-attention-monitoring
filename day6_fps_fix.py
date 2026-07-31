import cv2
import mediapipe as mp
import time
from ultralytics import YOLO

# Initialize both
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
yolo_model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

# FPS settings
prev_time = 0

# Frame skip settings
frame_count = 0
PROCESS_EVERY = 3  # only process every 2nd frame

# Store last results so screen doesnt flicker
last_student_count = 0
last_face_count = 0
last_boxes = []
last_detections = []

face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Only run detection every Nth frame
    if frame_count % PROCESS_EVERY == 0:

        # --- YOLOv8 ---
        yolo_results = yolo_model(
            frame, classes=[0], verbose=False)
        last_boxes = []
        last_student_count = 0
        for result in yolo_results:
            last_student_count = len(result.boxes)
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                last_boxes.append((x1, y1, x2, y2, conf))

        # --- MediaPipe ---
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB)
        face_results = face_detector.process(rgb_frame)
        last_detections = []
        last_face_count = 0
        if face_results.detections:
            last_face_count = len(face_results.detections)
            last_detections = face_results.detections

    # Always draw last known results
    # Draw YOLO boxes
    for (x1, y1, x2, y2, conf) in last_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                     (255, 0, 0), 2)
        cv2.putText(frame, f'{conf:.0%}',
                   (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 0, 0), 2)

    # Draw MediaPipe faces
    for detection in last_detections:
        mp_draw.draw_detection(frame, detection)

    # --- Info Panel ---
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

    cv2.putText(frame, f'Students   : {last_student_count}',
                (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 0), 1)

    cv2.putText(frame, f'Faces      : {last_face_count}',
                (10, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 1)

    cv2.imshow('Classroom Monitoring System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

face_detector.close()
cap.release()
cv2.destroyAllWindows()