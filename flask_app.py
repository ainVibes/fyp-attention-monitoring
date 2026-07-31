import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from flask import Flask, Response, render_template, jsonify, request
from ultralytics import YOLO
from picamera2 import Picamera2

# ── Runtime state: yawn / occlusion detection ─────────────────────────────────
occ_start_time    = None   # when the current hand-near-mouth occlusion began
mar_yawn_start     = None   # when the current open-mouth (unoccluded) MAR spike began
last_mar           = 0.0    # most recent reliable (unoccluded) MAR reading
occ_had_mar_rise   = False  # was the mouth already open the instant before this occlusion started?
squint_start = None   # when continuous squinting (while occluded) began
OCC_SQUINT_EAR_THRESH = 0.30   # looser than EAR_THRESH (blink cutoff) — catches squinting, not full blinks

# ── Runtime state: sustained low-attention tracking ───────────────────────────
low_attn_start        = None
sustained_alert_fired = False

# ── Yawn detection tuning ──────────────────────────────────────────────────────
MAR_YAWN_SECONDS     = 0.8   # mouth must stay open past MAR_THRESH this long to count as yawning
PRE_YAWN_MAR_THRESH  = 0.35  # "mouth already open" cutoff, checked at the moment hand covers it
OCC_YAWN_MIN_SECONDS = 0.6   # hand must stay on mouth at least this long to count
OCC_YAWN_MAX_SECONDS = 3.0   # beyond this, it's resting/thinking, not a yawn

# ── Sustained alert config ─────────────────────────────────────────────────────
SUSTAINED_ALERT_SECONDS = 15   # e.g. 5 min in production — tune this, it's just an example

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Init models ────────────────────────────────────────────────────────────────
mp_face_mesh  = mp.solutions.face_mesh
mp_pose       = mp.solutions.pose
mp_hands      = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
yolo_model    = YOLO('yolov8n.pt')

# ── Camera setup ───────────────────────────────────────────────────────────────
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "XBGR8888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

_init_frame = None
for attempt in range(10):
    _init_frame = picam2.capture_array()
    if _init_frame is not None:
        print(f"✓ Got valid frame on attempt {attempt+1}")
        break
    print(f"Attempt {attempt+1}: frame is None, retrying...")
    time.sleep(0.5)

if _init_frame is None:
    raise RuntimeError("Camera failed to produce a frame after 10 attempts")

_init_frame = _init_frame[:,:,2::-1].copy()
h, w = _init_frame.shape[:2]

# ── Camera matrix ──────────────────────────────────────────────────────────────
focal_length = w
cam_matrix   = np.array([[focal_length, 0, w/2],
                          [0, focal_length, h/2],
                          [0, 0, 1]], dtype=np.float64)
dist_coeffs  = np.zeros((4, 1))

MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0  ),
    (0.0,  -330.0, -65.0  ),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)
LM_IDS = [1, 152, 33, 263, 61, 291]

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_EAR_POSE  = 7
RIGHT_EAR_POSE = 8
LEFT_HIP       = 23
RIGHT_HIP      = 24

# ── Thresholds ─────────────────────────────────────────────────────────────────
YAW_THRESH        = 30
PITCH_THRESH      = 20
EAR_THRESH        = 0.25
BLINK_FRAMES      = 2
DROWSY_BLINK      = 15
MAR_THRESH        = 0.6
SHOULDER_TILT_MAX = 15
SPINE_LEAN_MAX    = 20
ATTENTION_THRESH  = 70

# ── Shared state ───────────────────────────────────────────────────────────────
state = {
    'sustained_alert'        : False,
    'low_attention_duration' : 0,     # seconds, for a live countdown on the dashboard
    'notifications'          : [],    # list of {message, timestamp}
    'student_count'  : 0,
    'face_count'     : 0,
    'attentive_count': 0,
    'attention_pct'  : 0,
    'alert'          : False,
    'blinks_per_min' : 0,
    'yawning_count'  : 0,
    'posture_status' : 'Unknown',
    'shoulder_tilt'  : 0.0,
    'spine_angle'    : 0.0,
    'fps'            : 0,
    'faces'          : [],
    'history'        : [],
}
state_lock = threading.Lock()

# ── Session control state ──────────────────────────────────────────────────────
session = {
    'active'   : False,
    'end_time' : None,   # epoch seconds, None = no timer (manual stop only)
}
session_lock = threading.Lock()

# ── Runtime globals ────────────────────────────────────────────────────────────
blink_counter    = 0
total_blinks     = 0
blink_start_time = time.time()
blinks_per_min   = 0
total_yawns      = 0
frame_count      = 0
PROCESS_EVERY    = 5
output_frame     = None
frame_lock       = threading.Lock()

face_mesh     = mp_face_mesh.FaceMesh(
    max_num_faces=5, refine_landmarks=False,
    min_detection_confidence=0.3, min_tracking_confidence=0.3)
pose_detector = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ── Helper functions ───────────────────────────────────────────────────────────
def ear(lm, ids, shape):
    fh, fw = shape[:2]
    pts = np.array([(lm[i].x*fw, lm[i].y*fh) for i in ids])
    v1  = np.linalg.norm(pts[1]-pts[5])
    v2  = np.linalg.norm(pts[2]-pts[4])
    h1  = np.linalg.norm(pts[0]-pts[3])
    return (v1+v2)/(2.0*h1+1e-6)

def hand_near_mouth(hands_res, mouth_center, shape, threshold=70):
    """Distance from the nearest hand landmark to the mouth center."""
    if not hands_res.multi_hand_landmarks:
        return False
    fh, fw = shape[:2]
    min_dist = float('inf')
    for hand_lms in hands_res.multi_hand_landmarks:
        for lm in hand_lms.landmark:
            hx, hy = lm.x*fw, lm.y*fh
            d = np.linalg.norm([hx-mouth_center[0], hy-mouth_center[1]])
            min_dist = min(min_dist, d)
    return min_dist < threshold

def mar(lm, shape):
    fh, fw = shape[:2]
    lc = np.array([lm[78].x*fw,  lm[78].y*fh])
    rc = np.array([lm[308].x*fw, lm[308].y*fh])
    pairs = [(13,14),(81,178),(311,402)]
    vsum  = sum(np.linalg.norm(
        np.array([lm[a].x*fw, lm[a].y*fh]) -
        np.array([lm[b].x*fw, lm[b].y*fh])) for a,b in pairs)
    return vsum / (2.0*np.linalg.norm(lc-rc)+1e-6)

def head_pose(lm, shape):
    fh, fw = shape[:2]
    img_pts = np.array([(lm[i].x*fw, lm[i].y*fh) for i in LM_IDS], dtype=np.float64)
    ok, rv, _ = cv2.solvePnP(MODEL_POINTS, img_pts, cam_matrix, dist_coeffs,
                              flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, None
    rm, _ = cv2.Rodrigues(rv)
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

def draw_axes(frame, lm, shape):
    fh, fw = shape[:2]
    img_pts = np.array([(lm[i].x*fw, lm[i].y*fh) for i in LM_IDS], dtype=np.float64)
    _, rv, tv = cv2.solvePnP(MODEL_POINTS, img_pts, cam_matrix, dist_coeffs,
                              flags=cv2.SOLVEPNP_ITERATIVE)
    nose = (int(img_pts[0][0]), int(img_pts[0][1]))
    proj, _ = cv2.projectPoints(np.float32([[60,0,0],[0,60,0],[0,0,60]]),
                                 rv, tv, cam_matrix, dist_coeffs)
    cv2.arrowedLine(frame, nose, tuple(proj[0].ravel().astype(int)), (0,0,255),   2, tipLength=0.2)
    cv2.arrowedLine(frame, nose, tuple(proj[1].ravel().astype(int)), (0,255,0),   2, tipLength=0.2)
    cv2.arrowedLine(frame, nose, tuple(proj[2].ravel().astype(int)), (255,120,0), 2, tipLength=0.2)

def analyse_posture(pose_lms, shape):
    fh, fw = shape[:2]
    def pt(idx):
        lm = pose_lms.landmark[idx]
        return np.array([lm.x*fw, lm.y*fh])
    ls  = pt(LEFT_SHOULDER);  rs  = pt(RIGHT_SHOULDER)
    lh  = pt(LEFT_HIP);       rh  = pt(RIGHT_HIP)
    le  = pt(LEFT_EAR_POSE);  re  = pt(RIGHT_EAR_POSE)
    mid_s = (ls+rs)/2;  mid_h = (lh+rh)/2;  mid_e = (le+re)/2
    dx = rs[0]-ls[0];   dy = rs[1]-ls[1]
    tilt = abs(np.degrees(np.arctan2(dy, dx)))
    if tilt > 90:
        tilt = 180 - tilt
    sv   = mid_s - mid_h
    lean = abs(np.degrees(np.arctan2(sv[0], -sv[1])))
    ok   = tilt < SHOULDER_TILT_MAX and lean < SPINE_LEAN_MAX
    fb   = ("Upright" if ok else
            (f"Side slouch {tilt:.0f}°" if tilt >= SHOULDER_TILT_MAX
             else f"Leaning {lean:.0f}°"))
    return tilt, lean, ok, fb, mid_s, mid_h, mid_e, ls, rs

def draw_posture_lines(frame, ms, mh, me, ls, rs, ok):
    c = (0,220,100) if ok else (0,80,255)
    cv2.line(frame, tuple(ls.astype(int)), tuple(rs.astype(int)), c, 2)
    cv2.line(frame, tuple(mh.astype(int)), tuple(ms.astype(int)), c, 2)
    cv2.line(frame, tuple(ms.astype(int)), tuple(me.astype(int)), c, 2)
    for p in [mh, ms, me, ls, rs]:
        cv2.circle(frame, tuple(p.astype(int)), 4, c, -1)

def attention_score(pose_ok, drowsy, yawning, posture_ok):
    score = 0
    if pose_ok:     score += 25
    if not drowsy:  score += 25
    if not yawning: score += 25
    if posture_ok:  score += 25
    return score

# ── Detection thread ────────────────────────────────────────────────────────────
def detection_loop():
    global blink_counter, total_blinks, blink_start_time, blinks_per_min
    global total_yawns, frame_count, output_frame
    global occ_start_time, occ_had_mar_rise, mar_yawn_start, last_mar
    global low_attn_start, sustained_alert_fired
    global squint_start

    prev_time        = time.time()
    history_timer    = time.time()
    last_boxes       = []
    last_pose_data   = []
    posture_ok       = True
    posture_feedback = 'No body'
    shoulder_tilt    = 0.0
    spine_angle      = 0.0

    while True:
        with session_lock:
            active   = session['active']
            end_time = session['end_time']

        if not active:
            idle = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(idle, 'Monitoring stopped - press Start', (60, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            _, buffer = cv2.imencode('.jpg', idle)
            with frame_lock:
                output_frame = buffer.tobytes()
            time.sleep(0.2)
            continue

        if end_time and time.time() >= end_time:
            with session_lock:
                session['active'] = False
            continue

        raw   = picam2.capture_array()
        frame = raw[:,:,2::-1].copy()

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
            results = yolo_model(frame, classes=[0], verbose=False)
            last_boxes    = []
            student_count = 0
            for r in results:
                student_count = len(r.boxes)
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    last_boxes.append((x1,y1,x2,y2,float(box.conf[0])))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_res = pose_detector.process(rgb)
            if pose_res.pose_landmarks:
                shoulder_tilt, spine_angle, posture_ok, posture_feedback, \
                ms, mh, me, ls, rs = analyse_posture(pose_res.pose_landmarks, frame.shape)
                draw_posture_lines(frame, ms, mh, me, ls, rs, posture_ok)

            face_res  = face_mesh.process(rgb)
            hands_res = hands_detector.process(rgb)

            last_pose_data = []
            if face_res.multi_face_landmarks:
                for face_lms in face_res.multi_face_landmarks:
                    lm = face_lms.landmark

                    yaw, pitch, roll = head_pose(lm, frame.shape)
                    pose_ok_face = (yaw is not None and
                                    abs(yaw) < YAW_THRESH and
                                    abs(pitch) < PITCH_THRESH)
                    draw_axes(frame, lm, frame.shape)

                    avg_ear = (ear(lm, LEFT_EYE,  frame.shape) +
                               ear(lm, RIGHT_EYE, frame.shape)) / 2.0
                    eye_col = (0,255,0) if avg_ear >= EAR_THRESH else (0,0,255)
                    for ids in [LEFT_EYE, RIGHT_EYE]:
                        pts = [(int(lm[i].x*frame.shape[1]),
                                int(lm[i].y*frame.shape[0])) for i in ids]
                        cv2.polylines(frame, [cv2.convexHull(np.array(pts))], True, eye_col, 1)

                    if avg_ear < EAR_THRESH:
                        blink_counter += 1
                    else:
                        if blink_counter >= BLINK_FRAMES:
                            total_blinks += 1
                        blink_counter = 0
                    drowsy = blinks_per_min > DROWSY_BLINK or avg_ear < (EAR_THRESH - 0.05)

                    # ── Yawn / occlusion detection ──────────────────────────
                    m = mar(lm, frame.shape)
                    mouth_center = (lm[13].x*frame.shape[1], lm[13].y*frame.shape[0])
                    occluded = hand_near_mouth(hands_res, mouth_center, frame.shape)

                    if occluded:
                        if occ_start_time is None:
                            occ_start_time   = curr_time
                            occ_had_mar_rise = last_mar > PRE_YAWN_MAR_THRESH
                        if avg_ear < OCC_SQUINT_EAR_THRESH:
                            if squint_start is None:
                                squint_start = curr_time
                        else:
                            squint_start = None
                        mar_yawn_start = None
                    else:
                        occ_start_time   = None
                        occ_had_mar_rise = False
                        squint_start     = None
                        last_mar = m

                        if m > MAR_THRESH:
                            if mar_yawn_start is None:
                                mar_yawn_start = curr_time
                        else:
                            if mar_yawn_start is not None:
                                total_yawns += 1
                            mar_yawn_start = None

                    occ_duration      = (curr_time - occ_start_time) if occ_start_time is not None else 0
                    mar_yawn_duration = (curr_time - mar_yawn_start) if mar_yawn_start is not None else 0

                    yawn_by_mar  = mar_yawn_duration >= MAR_YAWN_SECONDS
                    squint_duration = (curr_time - squint_start) if squint_start is not None else 0
                    yawn_by_hand = (
                        OCC_YAWN_MIN_SECONDS <= occ_duration <= OCC_YAWN_MAX_SECONDS
                        and (occ_had_mar_rise or squint_duration >= OCC_YAWN_MIN_SECONDS)
                    )
                    yawning      = yawn_by_mar or yawn_by_hand

                    mouth_col = (0,220,255) if yawning else (180,180,180)
                    mouth_ids = [78,81,13,311,308,402,14,178]
                    mpts = np.array([(int(lm[i].x*frame.shape[1]),
                                      int(lm[i].y*frame.shape[0])) for i in mouth_ids])
                    cv2.polylines(frame, [cv2.convexHull(mpts)], True, mouth_col, 1)

                    score     = attention_score(pose_ok_face, drowsy, yawning, posture_ok)
                    attentive = score >= 75

                    nx = int(lm[1].x*frame.shape[1])
                    ny = int(lm[1].y*frame.shape[0])

                    if yawn_by_hand:
                        col, status = (0,220,255), 'YAWNING (covered)'
                    elif attentive:
                        col, status = (0,220,80),   'ATTENTIVE'
                    elif yawn_by_mar:
                        col, status = (0,220,255),  'YAWNING'
                    elif drowsy:
                        col, status = (0,165,255),  'DROWSY'
                    elif not posture_ok:
                        col, status = (180,60,255), 'BAD POSTURE'
                    elif occluded:
                        col, status = (150,150,150), 'HAND NEAR MOUTH'
                    else:
                        col, status = (0,60,220),   'DISTRACTED'

                    cv2.putText(frame, status, (nx-45, ny-55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                    cv2.putText(frame, f'EAR:{avg_ear:.2f} MAR:{m:.2f}',
                                (nx-45, ny-35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

                    last_pose_data.append({
                        'status'    : status,
                        'attentive' : bool(attentive),
                        'score'     : int(score),
                        'yawning'   : bool(yawning),
                        'drowsy'    : bool(drowsy),
                        'posture_ok': bool(posture_ok),
                    })

            for (x1,y1,x2,y2,conf) in last_boxes:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,120,0), 2)

            face_total      = len(last_pose_data)
            attentive_count = sum(1 for f in last_pose_data if f['attentive'])
            attn_pct        = int(attentive_count/face_total*100) if face_total else 0
            alert           = attn_pct < ATTENTION_THRESH and face_total > 0
            yawning_count   = sum(1 for f in last_pose_data if f['yawning'])

            # ── Sustained low-attention tracking ─────────────────────────────
            if alert:
                if low_attn_start is None:
                    low_attn_start = curr_time
                low_attn_duration = curr_time - low_attn_start

                if low_attn_duration >= SUSTAINED_ALERT_SECONDS and not sustained_alert_fired:
                    sustained_alert_fired = True
                    with state_lock:
                        state['notifications'].append({
                            'message'  : f'Class attention has been below {ATTENTION_THRESH}% '
                                         f'for over {int(SUSTAINED_ALERT_SECONDS/60)} min. '
                                         f'Consider a break or activity change.',
                            'timestamp': curr_time,
                        })
                        if len(state['notifications']) > 10:
                            state['notifications'].pop(0)
            else:
                low_attn_start = None
                sustained_alert_fired = False
                low_attn_duration = 0

            with state_lock:
                state['sustained_alert']        = sustained_alert_fired
                state['low_attention_duration'] = int(low_attn_duration)
                state['student_count']   = student_count
                state['face_count']      = face_total
                state['attentive_count'] = attentive_count
                state['attention_pct']   = attn_pct
                state['alert']           = alert
                state['blinks_per_min']  = blinks_per_min
                state['yawning_count']   = yawning_count
                state['posture_status']  = posture_feedback
                state['shoulder_tilt']   = round(shoulder_tilt, 1)
                state['spine_angle']     = round(spine_angle, 1)
                state['fps']             = int(fps)
                state['faces']           = last_pose_data

                if curr_time - history_timer >= 3:
                    state['history'].append(attn_pct)
                    if len(state['history']) > 20:
                        state['history'].pop(0)
                    history_timer = curr_time

            alert_col = (0,60,220) if alert else (0,180,80)
            cv2.rectangle(frame, (5,5), (320,130), (0,0,0), -1)
            cv2.putText(frame, 'CLASSROOM MONITOR', (10,22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(frame, f'FPS: {int(fps)}  Students: {student_count}  Faces: {face_total}',
                        (10,44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
            cv2.putText(frame, f'Attentive: {attentive_count}/{face_total} ({attn_pct}%)',
                        (10,66), cv2.FONT_HERSHEY_SIMPLEX, 0.50, alert_col, 1)
            cv2.putText(frame, f'Blinks/min: {blinks_per_min}  Yawning: {yawning_count}',
                        (10,88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,165,255), 1)
            cv2.putText(frame, f'Posture: {posture_feedback}',
                        (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,60,255), 1)

            if alert:
                cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,180), 4)
                cv2.putText(frame, '! LOW ATTENTION ALERT !',
                            (frame.shape[1]//2-160, frame.shape[0]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            output_frame = buffer.tobytes()

        time.sleep(0.01)

# ── Flask routes ─────────────────────────────────────────────────────────────
def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.05)
                continue
            frame_bytes = output_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with state_lock:
        return jsonify(state)

@app.route('/api/start', methods=['POST'])
def start_session():
    data = request.get_json(silent=True) or {}
    duration_min = data.get('duration_minutes')
    with session_lock:
        session['active']   = True
        session['end_time'] = time.time() + duration_min * 60 if duration_min else None
    return jsonify({'status': 'started', 'end_time': session['end_time']})

@app.route('/api/stop', methods=['POST'])
def stop_session():
    with session_lock:
        session['active']   = False
        session['end_time'] = None
    return jsonify({'status': 'stopped'})

@app.route('/api/session_status')
def session_status():
    with session_lock:
        return jsonify(session)

# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    print("\n✓ Dashboard running at: http://127.0.0.1:5000")
    print("  Open this URL in your browser\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)