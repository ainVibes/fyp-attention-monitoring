import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO

# ── Init ──────────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_pose      = mp.solutions.pose
mp_draw      = mp.solutions.drawing_utils
yolo_model   = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
h, w = frame.shape[:2]
focal_length = w
cam_matrix  = np.array([[focal_length, 0,            w / 2],
                         [0,            focal_length, h / 2],
                         [0,            0,            1    ]], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))

# ── 3D model points ───────────────────────────────────────────────────────────
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0  ),
    (0.0,  -330.0, -65.0  ),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)
LM_IDS = [1, 152, 33, 263, 61, 291]

# ── Landmark indices ───────────────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# MediaPipe Pose landmark indices
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_EAR_POSE  = 7
RIGHT_EAR_POSE = 8
LEFT_HIP       = 23
RIGHT_HIP      = 24
NOSE_POSE      = 0

# ── Thresholds ────────────────────────────────────────────────────────────────
YAW_THRESH        = 30
PITCH_THRESH      = 20
EAR_THRESH        = 0.25
BLINK_FRAMES      = 2
DROWSY_BLINK      = 15
MAR_THRESH        = 0.6
YAWN_FRAMES       = 8
SHOULDER_TILT_MAX = 15    # degrees — shoulder level difference
SPINE_LEAN_MAX    = 20    # degrees — forward/backward lean from vertical

# ── Runtime state ─────────────────────────────────────────────────────────────
prev_time         = 0
frame_count       = 0
PROCESS_EVERY     = 3

last_student_count = 0
last_boxes         = []
last_pose_data     = []

blink_counter     = 0
total_blinks      = 0
blink_start_time  = time.time()
blinks_per_min    = 0
yawn_counter      = 0
total_yawns       = 0

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces            = 10,
    refine_landmarks         = False,
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5)

pose_detector = mp_pose.Pose(
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5)

# ── Helpers ───────────────────────────────────────────────────────────────────

def eye_aspect_ratio(landmarks, eye_indices, frame_shape):
    fh, fw = frame_shape[:2]
    pts = np.array([(landmarks[i].x * fw, landmarks[i].y * fh) for i in eye_indices])
    v1  = np.linalg.norm(pts[1] - pts[5])
    v2  = np.linalg.norm(pts[2] - pts[4])
    h1  = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h1 + 1e-6)


def mouth_aspect_ratio(landmarks, frame_shape):
    fh, fw = frame_shape[:2]
    lc  = np.array([landmarks[78 ].x * fw, landmarks[78 ].y * fh])
    rc  = np.array([landmarks[308].x * fw, landmarks[308].y * fh])
    t1  = np.array([landmarks[13 ].x * fw, landmarks[13 ].y * fh])
    b1  = np.array([landmarks[14 ].x * fw, landmarks[14 ].y * fh])
    t2  = np.array([landmarks[81 ].x * fw, landmarks[81 ].y * fh])
    b2  = np.array([landmarks[178].x * fw, landmarks[178].y * fh])
    t3  = np.array([landmarks[311].x * fw, landmarks[311].y * fh])
    b3  = np.array([landmarks[402].x * fw, landmarks[402].y * fh])
    return (np.linalg.norm(t1-b1) + np.linalg.norm(t2-b2) + np.linalg.norm(t3-b3)) / \
           (2.0 * np.linalg.norm(lc-rc) + 1e-6)


def get_head_pose(landmarks, frame_shape):
    fh, fw = frame_shape[:2]
    img_pts = np.array([
        (landmarks[i].x * fw, landmarks[i].y * fh) for i in LM_IDS
    ], dtype=np.float64)
    ok, rot_vec, _ = cv2.solvePnP(MODEL_POINTS, img_pts, cam_matrix, dist_coeffs,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, None
    rm, _ = cv2.Rodrigues(rot_vec)
    sy = np.sqrt(rm[0,0]**2 + rm[1,0]**2)
    if sy >= 1e-6:
        pitch = np.degrees(np.arctan2(-rm[2,0], sy))
        yaw   = np.degrees(np.arctan2( rm[1,0], rm[0,0]))
        roll  = np.degrees(np.arctan2( rm[2,1], rm[2,2]))
    else:
        pitch = np.degrees(np.arctan2(-rm[2,0], sy))
        yaw   = 0.0
        roll  = np.degrees(np.arctan2(-rm[1,2], rm[1,1]))
    return yaw, pitch, roll


def draw_axes(frame, landmarks, frame_shape):
    fh, fw = frame_shape[:2]
    img_pts = np.array([
        (landmarks[i].x * fw, landmarks[i].y * fh) for i in LM_IDS
    ], dtype=np.float64)
    _, rv, tv = cv2.solvePnP(MODEL_POINTS, img_pts, cam_matrix, dist_coeffs,
                              flags=cv2.SOLVEPNP_ITERATIVE)
    nose = (int(img_pts[0][0]), int(img_pts[0][1]))
    proj, _ = cv2.projectPoints(np.float32([[80,0,0],[0,80,0],[0,0,80]]),
                                 rv, tv, cam_matrix, dist_coeffs)
    cv2.arrowedLine(frame, nose, tuple(proj[0].ravel().astype(int)), (0,0,255),   2, tipLength=0.2)
    cv2.arrowedLine(frame, nose, tuple(proj[1].ravel().astype(int)), (0,255,0),   2, tipLength=0.2)
    cv2.arrowedLine(frame, nose, tuple(proj[2].ravel().astype(int)), (255,0,0),   2, tipLength=0.2)


def draw_eye_landmarks(frame, landmarks, eye_indices, frame_shape, color):
    fh, fw = frame_shape[:2]
    pts = [(int(landmarks[i].x * fw), int(landmarks[i].y * fh)) for i in eye_indices]
    for pt in pts:
        cv2.circle(frame, pt, 2, color, -1)
    cv2.polylines(frame, [cv2.convexHull(np.array(pts))], True, color, 1)


def draw_mouth_landmarks(frame, landmarks, frame_shape, color):
    fh, fw   = frame_shape[:2]
    mouth_ids = [78, 81, 13, 311, 308, 402, 14, 178]
    pts = np.array([(int(landmarks[i].x * fw), int(landmarks[i].y * fh)) for i in mouth_ids])
    cv2.polylines(frame, [cv2.convexHull(pts)], True, color, 1)


def analyse_posture(pose_lms, frame_shape):
    """
    Returns (shoulder_tilt_deg, spine_angle_deg, posture_ok, feedback_str)

    shoulder_tilt : angle of the shoulder line from horizontal
                    0° = level shoulders, higher = tilted/slouching sideways
    spine_angle   : angle of the neck-to-hip line from vertical
                    0° = perfectly upright, higher = leaning forward/backward
    posture_ok    : True if both within threshold
    feedback      : short string for display
    """
    fh, fw = frame_shape[:2]

    def pt(idx):
        lm = pose_lms.landmark[idx]
        return np.array([lm.x * fw, lm.y * fh])

    ls  = pt(LEFT_SHOULDER)
    rs  = pt(RIGHT_SHOULDER)
    lh  = pt(LEFT_HIP)
    rh  = pt(RIGHT_HIP)
    le  = pt(LEFT_EAR_POSE)
    re  = pt(RIGHT_EAR_POSE)

    # Mid-points
    mid_shoulder = (ls + rs) / 2
    mid_hip      = (lh + rh) / 2
    mid_ear      = (le + re) / 2

    # ── Shoulder tilt ─────────────────────────────────────────────────────────
    # Angle of shoulder line from horizontal
    dx = rs[0] - ls[0]
    dy = rs[1] - ls[1]
    shoulder_tilt = abs(np.degrees(np.arctan2(dy, dx)))
    # Normalise to 0-90
    if shoulder_tilt > 90:
        shoulder_tilt = 180 - shoulder_tilt

    # ── Spine / torso lean ────────────────────────────────────────────────────
    # Vector from mid-hip to mid-shoulder, angle from vertical (upward = 0°)
    spine_vec = mid_shoulder - mid_hip
    spine_angle = abs(np.degrees(np.arctan2(spine_vec[0], -spine_vec[1])))

    # ── Neck lean ─────────────────────────────────────────────────────────────
    # Vector from mid-shoulder to mid-ear
    neck_vec   = mid_ear - mid_shoulder
    neck_angle = abs(np.degrees(np.arctan2(neck_vec[0], -neck_vec[1])))

    posture_ok = shoulder_tilt < SHOULDER_TILT_MAX and spine_angle < SPINE_LEAN_MAX

    if not posture_ok:
        if shoulder_tilt >= SHOULDER_TILT_MAX:
            feedback = f"Slouch side ({shoulder_tilt:.0f}°)"
        else:
            feedback = f"Lean fwd ({spine_angle:.0f}°)"
    else:
        feedback = "Upright"

    return shoulder_tilt, spine_angle, neck_angle, posture_ok, feedback, \
           mid_shoulder, mid_hip, mid_ear, ls, rs


def draw_posture_lines(frame, mid_shoulder, mid_hip, mid_ear, ls, rs,
                       posture_ok, shoulder_tilt, spine_angle):
    """Draw skeleton lines for shoulder + spine visualisation."""
    color_ok  = (0, 255, 150)
    color_bad = (0, 80, 255)
    color_s   = color_ok if shoulder_tilt < SHOULDER_TILT_MAX else color_bad
    color_sp  = color_ok if spine_angle   < SPINE_LEAN_MAX    else color_bad

    # Shoulder line
    cv2.line(frame,
             (int(ls[0]), int(ls[1])),
             (int(rs[0]), int(rs[1])),
             color_s, 2)
    # Spine line (hip → shoulder → ear)
    cv2.line(frame,
             (int(mid_hip[0]),      int(mid_hip[1])),
             (int(mid_shoulder[0]), int(mid_shoulder[1])),
             color_sp, 2)
    cv2.line(frame,
             (int(mid_shoulder[0]), int(mid_shoulder[1])),
             (int(mid_ear[0]),      int(mid_ear[1])),
             color_sp, 2)
    # Dots at key joints
    for pt in [mid_hip, mid_shoulder, mid_ear,
               (ls[0], ls[1]), (rs[0], rs[1])]:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 4,
                   color_ok if posture_ok else color_bad, -1)

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    posture_ok       = True
    posture_feedback = "No body detected"
    shoulder_tilt    = 0.0
    spine_angle      = 0.0

    frame_count += 1
    curr_time    = time.time()
    fps          = 1 / (curr_time - prev_time + 1e-9)
    prev_time    = curr_time

    elapsed = curr_time - blink_start_time
    if elapsed >= 5.0:
        blinks_per_min   = int(total_blinks * (60 / elapsed))
        total_blinks     = 0
        blink_start_time = curr_time

    if frame_count % PROCESS_EVERY == 0:

        # ── YOLO ─────────────────────────────────────────────────────────
        yolo_results = yolo_model(frame, classes=[0], verbose=False)
        last_boxes         = []
        last_student_count = 0
        for result in yolo_results:
            last_student_count = len(result.boxes)
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                last_boxes.append((x1, y1, x2, y2, float(box.conf[0])))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Face Mesh ────────────────────────────────────────────────────
        face_result = face_mesh.process(rgb)

        # ── Pose ─────────────────────────────────────────────────────────
        pose_result = pose_detector.process(rgb)

        last_pose_data = []

        # ── Posture (one person at a time from Pose) ──────────────────────
        posture_ok      = True
        posture_feedback= "No body detected"
        shoulder_tilt   = 0.0
        spine_angle     = 0.0

        if pose_result.pose_landmarks:
            shoulder_tilt, spine_angle, neck_angle, posture_ok, posture_feedback, \
            mid_shoulder, mid_hip, mid_ear, ls, rs = \
                analyse_posture(pose_result.pose_landmarks, frame.shape)

            draw_posture_lines(frame, mid_shoulder, mid_hip, mid_ear,
                               ls, rs, posture_ok, shoulder_tilt, spine_angle)

        # ── Face features ────────────────────────────────────────────────
        if face_result.multi_face_landmarks:
            for face_lms in face_result.multi_face_landmarks:
                lm = face_lms.landmark

                # Head pose
                yaw, pitch, roll = get_head_pose(lm, frame.shape)
                pose_head_ok = (yaw is not None and
                                abs(yaw) < YAW_THRESH and
                                abs(pitch) < PITCH_THRESH)
                draw_axes(frame, lm, frame.shape)

                # EAR
                avg_ear   = (eye_aspect_ratio(lm, LEFT_EYE,  frame.shape) +
                             eye_aspect_ratio(lm, RIGHT_EYE, frame.shape)) / 2.0
                eye_color = (0,255,0) if avg_ear >= EAR_THRESH else (0,0,255)
                draw_eye_landmarks(frame, lm, LEFT_EYE,  frame.shape, eye_color)
                draw_eye_landmarks(frame, lm, RIGHT_EYE, frame.shape, eye_color)

                if avg_ear < EAR_THRESH:
                    blink_counter += 1
                else:
                    if blink_counter >= BLINK_FRAMES:
                        total_blinks += 1
                    blink_counter = 0
                drowsy = blinks_per_min > DROWSY_BLINK or avg_ear < (EAR_THRESH - 0.05)

                # MAR
                mar = mouth_aspect_ratio(lm, frame.shape)
                if mar > MAR_THRESH:
                    yawn_counter += 1
                else:
                    if yawn_counter >= YAWN_FRAMES:
                        total_yawns += 1
                    yawn_counter = 0
                yawning = yawn_counter >= YAWN_FRAMES
                draw_mouth_landmarks(frame, lm, frame.shape,
                                     (0,220,255) if yawning else (200,200,200))

                # Combined attention — all 4 signals
                attentive = pose_head_ok and not drowsy and not yawning and posture_ok

                nose_x = int(lm[1].x * frame.shape[1])
                nose_y = int(lm[1].y * frame.shape[0])

                last_pose_data.append({
                    'yaw': yaw, 'pitch': pitch,
                    'pose_head_ok': pose_head_ok,
                    'avg_ear': avg_ear,   'drowsy': drowsy,
                    'mar': mar,           'yawning': yawning,
                    'posture_ok': posture_ok,
                    'posture_feedback': posture_feedback,
                    'attentive': attentive,
                    'nose': (nose_x, nose_y)
                })

    # ── Draw per-face labels ───────────────────────────────────────────────────
    for (x1, y1, x2, y2, conf) in last_boxes:
        cv2.rectangle(frame, (x1,y1),(x2,y2),(255,120,0),2)
        cv2.putText(frame, f'{conf:.0%}', (x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,120,0), 1)

    attentive_count = 0
    for pd in last_pose_data:
        if pd['yaw'] is None:
            continue
        if pd['attentive']:
            attentive_count += 1

        if pd['attentive']:
            color, status = (0, 220, 80),  'ATTENTIVE'
        elif pd['yawning']:
            color, status = (0, 220, 255), 'YAWNING'
        elif pd['drowsy']:
            color, status = (0, 165, 255), 'DROWSY'
        elif not pd['posture_ok']:
            color, status = (180, 60, 255),'BAD POSTURE'
        else:
            color, status = (0, 60, 220),  'DISTRACTED'

        nx, ny = pd['nose']
        cv2.putText(frame, status,
                    (nx-50, ny-68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, f"Y:{pd['yaw']:+.0f} P:{pd['pitch']:+.0f}",
                    (nx-45, ny-48), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220,220,220), 1)
        cv2.putText(frame, f"EAR:{pd['avg_ear']:.2f}  MAR:{pd['mar']:.2f}",
                    (nx-45, ny-32), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220,220,220), 1)
        cv2.putText(frame, f"Posture: {pd['posture_feedback']}",
                    (nx-45, ny-16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220,220,220), 1)

    # ── Info panel ────────────────────────────────────────────────────────────
    face_total  = len(last_pose_data)
    attn_pct    = (attentive_count / face_total * 100) if face_total else 0
    pct_color   = (0, int(attn_pct*2.55), int((100-attn_pct)*2.55))
    yawning_now = sum(1 for pd in last_pose_data if pd.get('yawning'))

    cv2.rectangle(frame, (5,5),(360,205),(0,0,0),-1)
    cv2.putText(frame, 'CLASSROOM MONITOR  v5.0',
                (10,24),  cv2.FONT_HERSHEY_SIMPLEX, 0.56,(255,255,255),1)
    cv2.putText(frame, f'FPS          : {int(fps)}',
                (10,46),  cv2.FONT_HERSHEY_SIMPLEX, 0.50,(0,255,0),    1)
    cv2.putText(frame, f'Students     : {last_student_count}',
                (10,66),  cv2.FONT_HERSHEY_SIMPLEX, 0.50,(255,255,0),  1)
    cv2.putText(frame, f'Faces        : {face_total}',
                (10,86),  cv2.FONT_HERSHEY_SIMPLEX, 0.50,(0,255,255),  1)
    cv2.putText(frame, f'Attentive    : {attentive_count}/{face_total} ({attn_pct:.0f}%)',
                (10,106), cv2.FONT_HERSHEY_SIMPLEX, 0.50, pct_color,   1)
    cv2.putText(frame, f'Blinks/min   : {blinks_per_min}',
                (10,126), cv2.FONT_HERSHEY_SIMPLEX, 0.50,(0,165,255),  1)
    cv2.putText(frame, f'Yawning      : {yawning_now} face(s)  |  Total: {total_yawns}',
                (10,146), cv2.FONT_HERSHEY_SIMPLEX, 0.50,(0,220,255),  1)
    cv2.putText(frame, f'Posture      : {posture_feedback}  |  Shldr:{shoulder_tilt:.0f} Spine:{spine_angle:.0f}',
                (10,166), cv2.FONT_HERSHEY_SIMPLEX, 0.50,(180,60,255), 1)
    cv2.putText(frame, f'Shldr<{SHOULDER_TILT_MAX}  Spine<{SPINE_LEAN_MAX}  MAR>{MAR_THRESH}  EAR<{EAR_THRESH}',
                (10,186), cv2.FONT_HERSHEY_SIMPLEX, 0.38,(160,160,160),1)

    cv2.imshow('Classroom Monitor  –  All 4 Features', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

face_mesh.close()
pose_detector.close()
cap.release()
cv2.destroyAllWindows()