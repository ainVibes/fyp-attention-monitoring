import cv2
import time

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # Display FPS on screen
    cv2.putText(frame, f'FPS: {int(fps)}', 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    
    cv2.imshow('Classroom Monitor', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()