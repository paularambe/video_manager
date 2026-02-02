#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import threading
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List, Tuple

import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)
MODEL_FILENAME = "gesture_recognizer.task"


def ensure_model() -> Path:
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / MODEL_FILENAME
    if model_path.exists():
        return model_path

    import urllib.request
    print(f"[INFO] Descargando modelo MediaPipe a {model_path} ...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("[INFO] Modelo descargado correctamente.")
    return model_path


def _angle_between(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-6) -> float:
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return float("nan")
    v1 /= n1
    v2 /= n2
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _euler_from_rotation_matrix(R: np.ndarray):
    R = np.asarray(R, dtype=float)
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return yaw, pitch, roll


def compute_hand_telemetry(landmarks) -> str:
    if landmarks is None:
        return "Sin mano detectada"

    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=float)
    if pts.shape[0] < 21:
        return "Mano detectada (landmarks incompletos)"

    wrist = pts[0]
    index_mcp = pts[5]
    pinky_mcp = pts[17]
    palm_center = (wrist + index_mcp + pinky_mcp) / 3.0

    def _normalize(v, eps=1e-6):
        n = np.linalg.norm(v)
        if n < eps:
            return np.array([1.0, 0.0, 0.0])
        return v / n

    x_axis = _normalize(index_mcp - pinky_mcp)
    y_axis = _normalize(palm_center - wrist)
    z_axis = _normalize(np.cross(x_axis, y_axis))
    x_axis = _normalize(x_axis - np.dot(x_axis, z_axis) * z_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))

    R = np.column_stack((x_axis, y_axis, z_axis))
    yaw, pitch, roll = _euler_from_rotation_matrix(R)

    angles: Dict[str, float] = {}

    TH_CMC, TH_MCP, TH_IP, TH_TIP = 1, 2, 3, 4
    angles["Thumb_MCP"] = _angle_between(pts[TH_CMC] - pts[TH_MCP], pts[TH_IP] - pts[TH_MCP])
    angles["Thumb_IP"] = _angle_between(pts[TH_MCP] - pts[TH_IP], pts[TH_TIP] - pts[TH_IP])

    def finger_angles(name_prefix, idx_mcp):
        mcp = pts[idx_mcp]
        pip = pts[idx_mcp + 1]
        dip = pts[idx_mcp + 2]
        tip = pts[idx_mcp + 3]
        v_palm = palm_center - mcp
        v_mcp_pip = pip - mcp
        angles[f"{name_prefix}_MCP"] = _angle_between(v_palm, v_mcp_pip)
        angles[f"{name_prefix}_PIP"] = _angle_between(mcp - pip, dip - pip)
        angles[f"{name_prefix}_DIP"] = _angle_between(pip - dip, tip - dip)

    finger_angles("Index", 5)
    finger_angles("Middle", 9)
    finger_angles("Ring", 13)
    finger_angles("Pinky", 17)

    def fmt(a):
        if a is None or math.isnan(a):
            return "--"
        return f"{a:5.1f}°"

    tlm_text = [
        "Muñeca (orientación aproximada)",
        f"  Yaw:   {fmt(math.degrees(yaw))}",
        f"  Pitch: {fmt(math.degrees(pitch))}",
        f"  Roll:  {fmt(math.degrees(roll))}",
        "",
        "Pulgar:",
        f"  MCP: {fmt(angles['Thumb_MCP'])}   IP: {fmt(angles['Thumb_IP'])}",
        "",
        "Índice:",
        f"  MCP: {fmt(angles['Index_MCP'])}  PIP: {fmt(angles['Index_PIP'])}  DIP: {fmt(angles['Index_DIP'])}",
        "Medio:",
        f"  MCP: {fmt(angles['Middle_MCP'])} PIP: {fmt(angles['Middle_PIP'])} DIP: {fmt(angles['Middle_DIP'])}",
        "Anular:",
        f"  MCP: {fmt(angles['Ring_MCP'])}   PIP: {fmt(angles['Ring_PIP'])}   DIP: {fmt(angles['Ring_DIP'])}",
        "Meñique:",
        f"  MCP: {fmt(angles['Pinky_MCP'])}  PIP: {fmt(angles['Pinky_PIP'])}  DIP: {fmt(angles['Pinky_DIP'])}",
    ]
    return "\n".join(tlm_text)


class CAlgGesture:
    """
    Algoritmo de gestos para UNA cámara (MediaPipe).
    Crea un recognizer local dentro del hilo (más seguro por thread-safety).
    """

    def __init__(self, cam_id: int, min_interval: float = 0.06):
        self.cam_id = int(cam_id)
        self.min_interval = float(min_interval)

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def map_gesture_name(eng_name: str) -> str:
        mapping = {
            "None": "Ninguno",
            "Open_Palm": "Mano abierta ✋",
            "Closed_Fist": "Puño cerrado ✊",
            "Pointing_Up": "Índice arriba ☝️",
            "Thumb_Up": "Pulgar arriba 👍",
            "Thumb_Down": "Pulgar abajo 👎",
            "Victory": "Victoria ✌️",
            "ILoveYou": "Te quiero 🤟",
        }
        return mapping.get(eng_name, eng_name)

    def start(
        self,
        get_frame: Callable[[], Optional[np.ndarray]],
        publish_result: Callable[[Dict[str, Any]], None],
    ):
        if self._thread is not None:
            return
        self._stop_evt.clear()

        def loop():
            # recognizer local en este hilo
            try:
                model_path = ensure_model()
                BaseOptions = mp_python.BaseOptions
                GestureRecognizer = mp_vision.GestureRecognizer
                GestureRecognizerOptions = mp_vision.GestureRecognizerOptions
                RunningMode = mp_vision.RunningMode

                base_options = BaseOptions(model_asset_path=str(model_path))
                options = GestureRecognizerOptions(
                    base_options=base_options,
                    running_mode=RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                recognizer = GestureRecognizer.create_from_options(options)
            except Exception as e:
                print(f"[GEST][cam{self.cam_id}] No pude crear recognizer:", e)
                return

            last_t = 0.0
            while not self._stop_evt.is_set():
                now = time.time()
                if (now - last_t) < self.min_interval:
                    time.sleep(0.005)
                    continue

                frame = get_frame()
                if frame is None:
                    time.sleep(0.02)
                    continue

                res = self.predict_once(recognizer, frame)
                publish_result(res)

                last_t = time.time()

            try:
                recognizer.close()
            except Exception:
                pass

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            try:
                self._thread.join(timeout=0.5)
            except Exception:
                pass
        self._thread = None

    def predict_once(self, recognizer, frame_bgr: np.ndarray) -> Dict[str, Any]:
        # MediaPipe espera RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)

        try:
            result: mp_vision.GestureRecognizerResult = recognizer.recognize_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"[GEST][cam{self.cam_id}] recognize_for_video error:", e)
            return {
                "has_hand": False,
                "label_text": "Gesto: --",
                "label_color": "#cccccc",
                "gesture_rows": [],
                "telemetry_text": "Sin mano detectada",
                "landmarks_norm": [],
            }

        label_text = "Gesto: --"
        label_color = "#cccccc"
        gesture_rows: List[Tuple[str, float]] = []
        telemetry_text = "Sin mano detectada"
        landmarks_norm: List[List[Tuple[float, float]]] = []

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                landmarks_norm.append([(lm.x, lm.y) for lm in hand_lms])

        tele_landmarks = None
        if getattr(result, "hand_world_landmarks", None) and len(result.hand_world_landmarks) > 0:
            tele_landmarks = result.hand_world_landmarks[0]
        elif result.hand_landmarks and len(result.hand_landmarks) > 0:
            tele_landmarks = result.hand_landmarks[0]

        telemetry_text = compute_hand_telemetry(tele_landmarks)

        if result.gestures:
            main_hand_gestures = result.gestures[0]
            if main_hand_gestures:
                top = main_hand_gestures[0]
                eng_name = top.category_name
                score = top.score
                esp_name = self.map_gesture_name(eng_name)
                label_text = f"Gesto: {esp_name} ({eng_name}, {score:.2f})"
                label_color = "#00cc66" if eng_name != "None" else "#cccccc"
                for g in main_hand_gestures:
                    gesture_rows.append((g.category_name, g.score))

        return {
            "has_hand": bool(landmarks_norm),
            "label_text": label_text,
            "label_color": label_color,
            "gesture_rows": gesture_rows,
            "telemetry_text": telemetry_text,
            "landmarks_norm": landmarks_norm,
        }
