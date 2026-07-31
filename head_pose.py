import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO

# ── Init ──────────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_draw      = mp.solutions.drawing_utils
yolo_model   = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

# ── Camera intrinsics (estimate – replace with calibration values if you have them) ──
ret, frame = cap.read()
h, w = frame.shape[:2]
focal_length = w                       # rough estimate: focal = frame width
cam_matrix   = np.array([[focal_length, 0,            w / 2],
                          [0,            focal_length, h / 2],
                          [0,            0,            1     ]], dtype=np.float64)
dist_coeffs  = np.zeros((4, 1))        # assume no lens distortion

# ── 3-D model points of key facial landmarks (in mm, canonical head) ──
# Indices match MediaPipe Face Mesh 468-landmark model
MODEL_POINTS = np.array([
    (0.0,   0.0,    0.0  ),   # Nose tip          – lm 1
    (0.0,  -330.0, -65.0 ),   # Chin              – lm 152
    (-225.0, 170.0, -135.0),  # Left eye corner   – lm 33
    (225.0,  170.0, -135.0),  # Right eye corner  – lm 263
    (-150.0,-150.0, -125.0),  # Left mouth corner – lm 61
    (150.0, -150.0, -125.0),  # Right mouth corner– lm 291
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
LM_IDS = [1, 152, 33, 263, 61, 291]

# ── Attention thresholds ───────────────────────────────────────────────────────
YAW_THRESH   = 30   # °  – head turned left / right
PITCH_THRESH = 20   # °  – head tilted up / down

# ── Runtime state ─────────────────────────────────────────────────────────────
prev_time          = 0
frame_count        = 0
PROCESS_EVERY      = 3          # skip frames for performance

last_student_count = 0
last_boxes         = []
last_pose_data     = []         # list of dicts per face: yaw, pitch, roll, attentive

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces      = 10,
    refine_landmarks   = False,
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_head_pose(landmarks, frame_shape):
    """
    Solve PnP from 6 face landmarks and return (yaw, pitch, roll) in degrees.
    yaw   : + = looking right,   - = looking left
    pitch : + = looking up,      - = looking down
    roll  : + = tilting right,   - = tilting left
    """
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

    # Decompose rotation matrix into Euler angles
    sy = np.sqrt(rot_mat[0,0]**2 + rot_mat[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.degrees(np.arctan2(-rot_mat[2,0], sy))
        yaw   = np.degrees(np.arctan2( rot_mat[1,0], rot_mat[0,0]))
        roll  = np.degrees(np.arctan2( rot_mat[2,1], rot_mat[2,2]))
    else:
        pitch = np.degrees(np.arctan2(-rot_mat[2,0], sy))
        yaw   = 0.0
        roll  = np.degrees(np.arctan2(-rot_mat[1,2], rot_mat[1,1]))

    return yaw, pitch, roll


def is_attentive(yaw, pitch):
    """Return True if head pose is within attention thresholds."""
    if yaw is None:
        return False
    return abs(yaw) < YAW_THRESH and abs(pitch) < PITCH_THRESH


def draw_axes(frame, landmarks, frame_shape):
    """
    Project X/Y/Z axes onto the frame to visualise head orientation.
    Red = X (roll), Green = Y (pitch), Blue = Z (yaw).
    """
    h, w = frame_shape[:2]
    image_points = np.array([
        (landmarks[lm_id].x * w, landmarks[lm_id].y * h)
        for lm_id in LM_IDS
    ], dtype=np.float64)

    _, rot_vec, trans_vec = cv2.solvePnP(
        MODEL_POINTS, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)

    nose_tip = (int(image_points[0][0]), int(image_points[0][1]))
    axis_len  = 80
    axes_3d   = np.float32([[axis_len, 0, 0],
                             [0, axis_len, 0],
                             [0, 0, axis_len]])
    proj, _   = cv2.projectPoints(axes_3d, rot_vec, trans_vec,
                                   cam_matrix, dist_coeffs)

    px = tuple(proj[0].ravel().astype(int))
    py = tuple(proj[1].ravel().astype(int))
    pz = tuple(proj[2].ravel().astype(int))

    cv2.arrowedLine(frame, nose_tip, px, (0, 0, 255), 2, tipLength=0.2)   # X – Red
    cv2.arrowedLine(frame, nose_tip, py, (0, 255, 0), 2, tipLength=0.2)   # Y – Green
    cv2.arrowedLine(frame, nose_tip, pz, (255, 0, 0), 2, tipLength=0.2)   # Z – Blue

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    curr_time   = time.time()
    fps         = 1 / (curr_time - prev_time + 1e-9)
    prev_time   = curr_time

    if frame_count % PROCESS_EVERY == 0:

        # ── YOLO person detection ────────────────────────────────────────
        yolo_results       = yolo_model(frame, classes=[0], verbose=False)
        last_boxes         = []
        last_student_count = 0
        for result in yolo_results:
            last_student_count = len(result.boxes)
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                last_boxes.append((x1, y1, x2, y2, conf))

        # ── MediaPipe head pose ──────────────────────────────────────────
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        last_pose_data = []

        if result.multi_face_landmarks:
            for face_lms in result.multi_face_landmarks:
                lm    = face_lms.landmark
                yaw, pitch, roll = get_head_pose(lm, frame.shape)
                attentive = is_attentive(yaw, pitch)
                draw_axes(frame, lm, frame.shape)

                # Nose tip pixel coords for label placement
                nose_x = int(lm[1].x * frame.shape[1])
                nose_y = int(lm[1].y * frame.shape[0])

                last_pose_data.append({
                    'yaw': yaw, 'pitch': pitch, 'roll': roll,
                    'attentive': attentive,
                    'nose': (nose_x, nose_y)
                })

    # ── Draw everything ────────────────────────────────────────────────────────

    # YOLO bounding boxes
    for (x1, y1, x2, y2, conf) in last_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
        cv2.putText(frame, f'{conf:.0%}', (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)

    # Per-face attention labels
    attentive_count = 0
    for pd in last_pose_data:
        if pd['yaw'] is None:
            continue
        if pd['attentive']:
            attentive_count += 1
        color  = (0, 220, 80) if pd['attentive'] else (0, 60, 220)
        status = 'ATTENTIVE' if pd['attentive'] else 'DISTRACTED'
        nx, ny = pd['nose']

        cv2.putText(frame, status, (nx - 40, ny - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame,
                    f"Y:{pd['yaw']:+.0f} P:{pd['pitch']:+.0f} R:{pd['roll']:+.0f}",
                    (nx - 55, ny - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    # ── Info panel ────────────────────────────────────────────────────────────
    face_total = len(last_pose_data)
    attn_pct   = (attentive_count / face_total * 100) if face_total else 0

    # Colour the attention % red → green
    pct_color = (0, int(attn_pct * 2.55), int((100 - attn_pct) * 2.55))

    cv2.rectangle(frame, (5, 5), (320, 140), (0, 0, 0), -1)

    cv2.putText(frame, 'CLASSROOM MONITOR  v2.0',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1)
    cv2.putText(frame, f'FPS        : {int(fps)}',
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1)
    cv2.putText(frame, f'Students   : {last_student_count}',
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)
    cv2.putText(frame, f'Faces      : {face_total}',
                (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
    cv2.putText(frame, f'Attentive  : {attentive_count}/{face_total}  ({attn_pct:.0f}%)',
                (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.52, pct_color, 1)
    cv2.putText(frame, f'Thresh Y<{YAW_THRESH} P<{PITCH_THRESH}',
                (10, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

    cv2.imshow('Classroom Monitoring System – Head Pose', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

face_mesh.close()
cap.release()
cv2.destroyAllWindows()