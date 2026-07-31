import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 nano model
# First run will auto download the model (~6MB)
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # YOLOv8 detection
    # classes=[0] means detect persons only
    results = model(frame, classes=[0], verbose=False)
    
    student_count = 0
    for result in results:
        student_count = len(result.boxes)
        
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw blue box around person
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                         (255, 0, 0), 2)
                         
            # Show confidence score
            cv2.putText(frame, f'Student {conf:.0%}',
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 0, 0), 2)
    
    # Info display
    cv2.putText(frame, f'FPS: {int(fps)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
                
    cv2.putText(frame, f'Students Detected: {student_count}',
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    
    cv2.imshow('Classroom Monitor - YOLOv8', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()