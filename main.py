import csv
import os
import time
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

CNN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eye_state_cnn.h5")

WINDOW_NAME = "Driver Drowsiness Detection (press q to quit)"


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data-user")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("กำลังดาวน์โหลดโมเดล face_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("ดาวน์โหลดเสร็จสิ้น")


CAMERA_INDEX = 0
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

EAR_THRESHOLD_DEFAULT = 0.21
MAR_THRESHOLD = 0.60

CNN_IMG_SIZE = 64
CNN_OPEN_THRESHOLD = 0.5
EAR_STRONG_CLOSED = 0.80
ENABLE_YAWN_ALERT = True
CALIB_SEC = 3.0
EYE_CROP_SCALE = 1.8
CLOSE_GRACE_SEC = 0.25

EYE_CLOSED_SOFT_SEC = 5.0
EYE_CLOSED_LOUD_SEC = 10.0
YAWN_SEC = 3.0

BLINK_MIN_SEC = 0.05
BLINK_MAX_SEC = 1.0

YAWN_COOLDOWN = 0.5
SOFT_COOLDOWN = 2.0
LOUD_COOLDOWN = 0.9


RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = dict(left=61, right=291, up_in=13, low_in=14,
             up_l=82, low_l=87, up_r=312, low_r=317)


def eye_aspect_ratio(pts):
    a = dist.euclidean(pts[1], pts[5])
    b = dist.euclidean(pts[2], pts[4])
    c = dist.euclidean(pts[0], pts[3])
    return (a + b) / (2.0 * c)


def mouth_aspect_ratio(lm):
    m = MOUTH
    v1 = dist.euclidean(lm[m["up_l"]], lm[m["low_l"]])
    v2 = dist.euclidean(lm[m["up_in"]], lm[m["low_in"]])
    v3 = dist.euclidean(lm[m["up_r"]], lm[m["low_r"]])
    h = dist.euclidean(lm[m["left"]], lm[m["right"]])
    return (v1 + v2 + v3) / (3.0 * h)


def crop_eye(gray, pts):
    h, w = gray.shape[:2]
    cx, cy = pts.mean(axis=0)
    half = int((pts[:, 0].max() - pts[:, 0].min()) * EYE_CROP_SCALE / 2)
    x1, y1 = max(0, int(cx) - half), max(0, int(cy) - half)
    x2, y2 = min(w, int(cx) + half), min(h, int(cy) + half)
    crop = gray[y1:y2, x1:x2]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    crop = cv2.equalizeHist(crop)
    crop = cv2.resize(crop, (CNN_IMG_SIZE, CNN_IMG_SIZE)).astype("float32") / 255.0
    return crop[..., None]


def fit_to_window(frame, window_name):
    try:
        _, _, ww, wh = cv2.getWindowImageRect(window_name)
    except cv2.error:
        return frame
    if ww <= 0 or wh <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = min(ww / w, wh / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((wh, ww, 3), dtype=frame.dtype)
    x0, y0 = (ww - nw) // 2, (wh - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


class Alarm:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
        self.soft = self._tone(660, 0.35, 0.25)
        self.loud = self._tone(1500, 0.6, 1.0)
        self.last_soft = 0.0
        self.last_loud = 0.0
        self.last_yawn = 0.0

    @staticmethod
    def _tone(freq, dur, volume):
        sr = 44100
        t = np.linspace(0, dur, int(sr * dur), False)
        wave = np.sin(2 * np.pi * freq * t)
        fade = np.minimum(1, np.minimum(t, dur - t) / 0.03)
        data = (wave * fade * volume * 32767).astype(np.int16)
        return pygame.mixer.Sound(buffer=data.tobytes())

    def play_yawn(self):
        now = time.time()
        if now - self.last_yawn >= YAWN_COOLDOWN:
            self.soft.play()
            self.last_yawn = now

    def play_soft(self):
        now = time.time()
        if now - self.last_soft >= SOFT_COOLDOWN:
            self.soft.play()
            self.last_soft = now

    def play_loud(self):
        now = time.time()
        if now - self.last_loud >= LOUD_COOLDOWN:
            self.loud.play()
            self.last_loud = now


class EventLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        name = os.path.join(LOG_DIR, datetime.now().strftime("drowsiness_log_%Y%m%d_%H%M%S.csv"))
        self.f = open(name, "w", newline="", encoding="utf-8-sig")
        self.w = csv.writer(self.f)
        self.w.writerow(["timestamp", "event", "level", "duration_sec", "detail"])
        self.name = name

    def log(self, event, level, duration=0.0, detail=""):
        self.w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         event, level, f"{duration:.2f}", detail])
        self.f.flush()
        print(f"[{level}] {event} ({duration:.1f}s) {detail}")

    def close(self):
        self.f.close()


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) if os.name == "nt" \
        else cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("เปิดกล้องไม่ได้ ตรวจสอบ CAMERA_INDEX")
        return

    ensure_model()

    cnn_model = None
    if os.path.exists(CNN_MODEL_PATH):
        cnn_model = load_model(CNN_MODEL_PATH)
        print("โหลด CNN Model สำเร็จแล้ว")
    else:
        print("คำเตือน: ไม่พบ eye_state_cnn.h5 (รัน notebook เพื่อเทรนก่อน) ตอนนี้ใช้ EAR สำรองไปก่อน")

    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5))

    alarm = Alarm()
    logger = EventLogger()

    blink_count = 0
    yawn_count = 0
    soft_count = 0
    loud_count = 0
    blink_times = deque()

    eye_start = None
    last_closed_t = 0.0
    yawn_start = None
    eye_soft_fired = False
    eye_loud_fired = False
    yawn_fired = False

    calibration_ears = []
    dynamic_ear_threshold = EAR_THRESHOLD_DEFAULT
    is_calibrated = False

    fps_t, fps = time.time(), 0.0
    session_start = time.time()
    calib_start = session_start
    frame_idx = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(mp_img, int(now * 1000))
            status_text, status_color = "Normal", (0, 200, 0)

            if res.face_landmarks:
                lm = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]])

                ear = (eye_aspect_ratio(lm[RIGHT_EYE]) + eye_aspect_ratio(lm[LEFT_EYE])) / 2.0
                mar = mouth_aspect_ratio(lm)

                if not is_calibrated:
                    calibration_ears.append(ear)
                    if now - calib_start >= CALIB_SEC:
                        open_ear = np.percentile(calibration_ears, 75)
                        dynamic_ear_threshold = min(0.25, max(0.12, open_ear * 0.65))
                        is_calibrated = True
                        print(f"Calibrated Threshold สำหรับตาคุณ: {dynamic_ear_threshold:.2f}")

                is_closed_ear = ear < dynamic_ear_threshold

                cnn_open_prob = None
                if cnn_model is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    crops = [crop_eye(gray, lm[RIGHT_EYE]), crop_eye(gray, lm[LEFT_EYE])]
                    crops = [c for c in crops if c is not None]
                    if crops:
                        probs = cnn_model(np.stack(crops), training=False).numpy().ravel()
                        cnn_open_prob = float(probs.mean())

                if not is_calibrated:
                    raw_closed = False
                elif not is_closed_ear:
                    raw_closed = False
                elif cnn_open_prob is None:
                    raw_closed = True
                else:
                    cnn_says_closed = cnn_open_prob < CNN_OPEN_THRESHOLD
                    strongly_closed = ear < dynamic_ear_threshold * EAR_STRONG_CLOSED
                    raw_closed = cnn_says_closed or strongly_closed

                if raw_closed:
                    last_closed_t = now

                eye_closed = raw_closed or (eye_start is not None and now - last_closed_t <= CLOSE_GRACE_SEC)
                mouth_open = mar > MAR_THRESHOLD

                if eye_closed:
                    if eye_start is None:
                        eye_start = now
                    eye_dur = now - eye_start

                    if eye_dur >= EYE_CLOSED_SOFT_SEC and not eye_soft_fired:
                        eye_soft_fired = True
                        soft_count += 1
                        logger.log("EYES_CLOSED_LONG", "SOFT", eye_dur, f"EAR={ear:.2f}")

                    if eye_dur >= EYE_CLOSED_LOUD_SEC:
                        if not eye_loud_fired:
                            eye_loud_fired = True
                            loud_count += 1
                            logger.log("EYES_CLOSED_VERY_LONG", "LOUD", eye_dur, f"EAR={ear:.2f}")
                        alarm.play_loud()
                    elif eye_soft_fired:
                        alarm.play_soft()
                else:
                    if eye_start is not None:
                        d = last_closed_t - eye_start
                        if BLINK_MIN_SEC <= d <= BLINK_MAX_SEC:
                            blink_count += 1
                            blink_times.append(now)
                    eye_start = None
                    eye_soft_fired = eye_loud_fired = False
                    eye_dur = 0.0

                if mouth_open:
                    if yawn_start is None:
                        yawn_start = now
                    yawn_dur = now - yawn_start
                    if yawn_dur >= YAWN_SEC:
                        if ENABLE_YAWN_ALERT:
                            alarm.play_yawn()
                    if yawn_dur >= YAWN_SEC and not yawn_fired:
                        yawn_fired = True
                        yawn_count += 1
                        if ENABLE_YAWN_ALERT:
                            soft_count += 1
                        logger.log("YAWN", "SOFT" if ENABLE_YAWN_ALERT else "INFO", yawn_dur, f"MAR={mar:.2f}")
                else:
                    yawn_start, yawn_fired = None, False

                if eye_loud_fired:
                    status_text, status_color = "DANGER! WAKE UP!", (0, 0, 255)
                elif eye_soft_fired or (ENABLE_YAWN_ALERT and yawn_fired):
                    status_text, status_color = "WARNING: Drowsy", (0, 165, 255)

                for idx in RIGHT_EYE + LEFT_EYE:
                    cv2.circle(frame, tuple(lm[idx].astype(int)), 2, (0, 255, 0), -1)
                for idx in MOUTH.values():
                    cv2.circle(frame, tuple(lm[idx].astype(int)), 2, (255, 200, 0), -1)
                x1, y1 = lm.min(axis=0).astype(int)
                x2, y2 = lm.max(axis=0).astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)

                while blink_times and now - blink_times[0] > 60:
                    blink_times.popleft()

                calib_str = f"Thresh: {dynamic_ear_threshold:.2f}" if is_calibrated else "Calibrating: keep eyes open"
                info = [
                    f"Eye: {'CLOSED' if eye_closed else 'Open'}  EAR={ear:.2f} ({calib_str})  "
                    + (f"CNN open={cnn_open_prob:.2f}" if cnn_open_prob is not None else "CNN: n/a"),
                    f"Closed for: {eye_dur:.1f}s  (5s=soft, 10s=LOUD)",
                    f"Mouth MAR={mar:.2f}  {'YAWNING' if mouth_open else ''}",
                    f"Blinks: {blink_count} ({len(blink_times)}/min)  Yawns: {yawn_count}",
                ]
                for i, t in enumerate(info):
                    cv2.putText(frame, t, (10, 55 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                eye_start = yawn_start = None
                eye_soft_fired = eye_loud_fired = yawn_fired = False
                status_text, status_color = "No face detected", (128, 128, 128)

            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - fps_t))
            fps_t = now
            cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if status_text.startswith("DANGER"):
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

            cv2.imshow(WINDOW_NAME, fit_to_window(frame, WINDOW_NAME))
            key = cv2.waitKey(1) & 0xFF
            frame_idx += 1

            if key == ord("q"):
                break
            if key == ord("c"):
                calibration_ears, is_calibrated = [], False
                calib_start = time.time()
                eye_start = None
                eye_soft_fired = eye_loud_fired = False
            if frame_idx > 3 and cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        total = time.time() - session_start
        logger.log("SESSION_SUMMARY", "INFO", total,
                   f"blinks={blink_count}, yawns={yawn_count}, "
                   f"soft_alerts={soft_count}, loud_alerts={loud_count}")
        logger.close()
        print(f"\nบันทึกไฟล์ log: {logger.name}")
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nหยุดโปรแกรม (Ctrl+C)")