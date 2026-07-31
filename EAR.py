import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO

# ── Init ──────────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
yolo_model   = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

# ── Camera intrinsics ─────────────────────────────────────────────────────────
ret, frame = cap.read()
h, w = frame.shape[:2]
focal_length = w
cam_matrix  = np.array([[focal_length, 0,            w / 2],
                         [0,            focal_length, h / 2],
                         [0,            0,            1    ]], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))

# ── 3D model points for head pose ─────────────────────────────────────────────
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0  ),   # Nose tip        – lm 1
    (0.0,  -330.0, -65.0  ),   # Chin            – lm 152
    (-225.0, 170.0, -135.0),   # Left eye corner – lm 33
    (225.0,  170.0, -135.0),   # Right eye corner– lm 263
    (-150.0,-150.0, -125.0),   # Left mouth      – lm 61
    (150.0, -150.0, -125.0),   # Right mouth     – lm 291
], dtype=np.float64)
LM_IDS = [1, 152, 33, 263, 61, 291]

# ── EAR landmark indices (MediaPipe Face Mesh) ────────────────────────────────
# Each eye needs 6 points: 2 horizontal + 4 vertical
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ── Thresholds ────────────────────────────────────────────────────────────────
YAW_THRESH   = 30    # degrees
PITCH_THRESH = 20    # degrees
EAR_THRESH   = 0.25  # below this = eye closing
BLINK_FRAMES = 2     # consecutive frames to confirm a blink
DROWSY_BLINK = 15    # blinks per minute above this = drowsy

# ── Runtime state ─────────────────────────────────────────────────────────────
prev_time          = 0
frame_count        = 0
PROCESS_EVERY      = 3

last_student_count = 0
last_boxes         = []
last_pose_data     = []

# Blink tracking (per-face — simplified for single face; extendable)
blink_counter      = 0      # frames eye has been below EAR threshold
total_blinks       = 0
blink_start_time   = time.time()
blinks_per_min     = 0

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces            = 10,
    refine_landmarks         = False,
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5)

# ── Helper functions ──────────────────────────────────────────────────────────

def eye_aspect_ratio(landmarks, eye_indices, frame_shape):
    """
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    p1,p4 = horizontal endpoints
    p2,p3,p5,p6 = vertical points
    Returns a float — lower means more closed.
    """
    h, w = frame_shape[:2]
    pts = np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in eye_indices
    ])
    # Vertical distances
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    h1 = np.linalg.norm(pts[0] - pts[3])
    ear = (v1 + v2) / (2.0 * h1 + 1e-6)
    return ear


def get_head_pose(landmarks, frame_shape):
    h, w = frame_shape[:2]
    image_points = np.array([
        (landmarks[lm_id].x * w, landmarks[lm_id].y * h)
        for lm_id in LM_IDS
    ], dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None, None, None

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    sy = np.sqrt(rot_mat[0,0]**2 + rot_mat[1,0]**2)
    if sy >= 1e-6:
        pitch = np.degrees(np.arctan2(-rot_mat[2,0], sy))
        yaw   = np.degrees(np.arctan2( rot_mat[1,0], rot_mat[0,0]))
        roll  = np.degrees(np.arctan2( rot_mat[2,1], rot_mat[2,2]))
    else:
        pitch = np.degrees(np.arctan2(-rot_mat[2,0], sy))
        yaw   = 0.0
        roll  = np.degrees(np.arctan2(-rot_mat[1,2], rot_mat[1,1]))
    return yaw, pitch, roll


def draw_axes(frame, landmarks, frame_shape):
    h, w = frame_shape[:2]
    image_points = np.array([
        (landmarks[lm_id].x * w, landmarks[lm_id].y * h)
        for lm_id in LM_IDS
    ], dtype=np.float64)
    _, rot_vec, trans_vec = cv2.solvePnP(
        MODEL_POINTS, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)
    nose_tip = (int(image_points[0][0]), int(image_points[0][1]))
    axes_3d  = np.float32([[80,0,0],[0,80,0],[0,0,80]])
    proj, _  = cv2.projectPoints(axes_3d, rot_vec, trans_vec,
                                  cam_matrix, dist_coeffs)
    cv2.arrowedLine(frame, nose_tip, tuple(proj[0].ravel().astype(int)), (0,0,255), 2, tipLength=0.2)
    cv2.arrowedLine(frame, nose_tip, tuple(proj[1].ravel().astype(int)), (0,255,0), 2, tipLength=0.2)
    cv2.arrowedLine(frame, nose_tip, tuple(proj[2].ravel().astype(int)), (255,0,0), 2, tipLength=0.2)


def draw_eye_landmarks(frame, landmarks, eye_indices, frame_shape, color):
    """Draw dots on the 6 eye landmark points."""
    h, w = frame_shape[:2]
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    for pt in pts:
        cv2.circle(frame, pt, 2, color, -1)
    # draw the eye outline
    hull = cv2.convexHull(np.array(pts))
    cv2.polylines(frame, [hull], True, color, 1)

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    curr_time    = time.time()
    fps          = 1 / (curr_time - prev_time + 1e-9)
    prev_time    = curr_time

    # Update blinks per minute every 5 seconds
    elapsed = curr_time - blink_start_time
    if elapsed >= 5.0:
        blinks_per_min    = int(total_blinks * (60 / elapsed))
        total_blinks      = 0
        blink_start_time  = curr_time

    if frame_count % PROCESS_EVERY == 0:

        # ── YOLO ─────────────────────────────────────────────────────────
        yolo_results       = yolo_model(frame, classes=[0], verbose=False)
        last_boxes         = []
        last_student_count = 0
        for result in yolo_results:
            last_student_count = len(result.boxes)
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                last_boxes.append((x1, y1, x2, y2, float(box.conf[0])))

        # ── MediaPipe Face Mesh ───────────────────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        last_pose_data = []

        if result.multi_face_landmarks:
            for face_lms in result.multi_face_landmarks:
                lm = face_lms.landmark

                # ── Head Pose ─────────────────────────────────────────────
                yaw, pitch, roll = get_head_pose(lm, frame.shape)
                pose_attentive   = (yaw is not None and
                                    abs(yaw) < YAW_THRESH and
                                    abs(pitch) < PITCH_THRESH)
                draw_axes(frame, lm, frame.shape)

                # ── EAR ───────────────────────────────────────────────────
                left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  frame.shape)
                right_ear = eye_aspect_ratio(lm, RIGHT_EYE, frame.shape)
                avg_ear   = (left_ear + right_ear) / 2.0

                # Draw eye contours — green if open, red if closing
                eye_color = (0, 255, 0) if avg_ear >= EAR_THRESH else (0, 0, 255)
                draw_eye_landmarks(frame, lm, LEFT_EYE,  frame.shape, eye_color)
                draw_eye_landmarks(frame, lm, RIGHT_EYE, frame.shape, eye_color)

                # Blink detection logic
                eye_closed = avg_ear < EAR_THRESH
                if eye_closed:
                    blink_counter += 1
                else:
                    if blink_counter >= BLINK_FRAMES:
                        total_blinks += 1
                    blink_counter = 0

                drowsy = blinks_per_min > DROWSY_BLINK or avg_ear < (EAR_THRESH - 0.05)

                # ── Combined attention ────────────────────────────────────
                attentive = pose_attentive and not drowsy

                # Nose tip for label placement
                nose_x = int(lm[1].x * frame.shape[1])
                nose_y = int(lm[1].y * frame.shape[0])

                last_pose_data.append({
                    'yaw': yaw, 'pitch': pitch, 'roll': roll,
                    'pose_attentive': pose_attentive,
                    'avg_ear': avg_ear,
                    'drowsy': drowsy,
                    'attentive': attentive,
                    'nose': (nose_x, nose_y)
                })

    # ── Draw ──────────────────────────────────────────────────────────────────

    # YOLO boxes
    for (x1, y1, x2, y2, conf) in last_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
        cv2.putText(frame, f'{conf:.0%}', (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)

    # Per-face labels
    attentive_count = 0
    for pd in last_pose_data:
        if pd['yaw'] is None:
            continue
        if pd['attentive']:
            attentive_count += 1

        # Status colour & text
        if pd['attentive']:
            color, status = (0, 220, 80), 'ATTENTIVE'
        elif pd['drowsy']:
            color, status = (0, 165, 255), 'DROWSY'
        else:
            color, status = (0, 60, 220), 'DISTRACTED'

        nx, ny = pd['nose']
        cv2.putText(frame, status,
                    (nx - 40, ny - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame,
                    f"Y:{pd['yaw']:+.0f} P:{pd['pitch']:+.0f}",
                    (nx - 45, ny - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        cv2.putText(frame,
                    f"EAR:{pd['avg_ear']:.2f}",
                    (nx - 25, ny - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    # ── Info panel ────────────────────────────────────────────────────────────
    face_total = len(last_pose_data)
    attn_pct   = (attentive_count / face_total * 100) if face_total else 0
    pct_color  = (0, int(attn_pct * 2.55), int((100 - attn_pct) * 2.55))

    cv2.rectangle(frame, (5, 5), (330, 165), (0, 0, 0), -1)

    cv2.putText(frame, 'CLASSROOM MONITOR  v3.0',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255,255,255), 1)
    cv2.putText(frame, f'FPS           : {int(fps)}',
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,255,0), 1)
    cv2.putText(frame, f'Students      : {last_student_count}',
                (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,0), 1)
    cv2.putText(frame, f'Faces         : {face_total}',
                (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,255,255), 1)
    cv2.putText(frame, f'Attentive     : {attentive_count}/{face_total} ({attn_pct:.0f}%)',
                (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.50, pct_color, 1)
    cv2.putText(frame, f'Blinks/min    : {blinks_per_min}',
                (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,165,255), 1)
    cv2.putText(frame, f'EAR thresh    : {EAR_THRESH}  |  Yaw<{YAW_THRESH} Pitch<{PITCH_THRESH}',
                (10, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)

    cv2.imshow('Classroom Monitor  –  Head Pose + EAR', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

face_mesh.close()
cap.release()
cv2.destroyAllWindows()