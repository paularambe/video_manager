#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import websockets

from CAlgFace import FaceEngine, CAlgFace, FaceResult


FACE_HOLD_SEC = 0.60
BBOX_EMA_ALPHA = 0.65


def jpeg_bytes_to_bgr(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    if not jpeg_bytes:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def draw_overlay_inplace(frame_bgr: np.ndarray, name: Optional[str], sim: float,
                         bbox_s: Optional[Tuple[float, float, float, float]]):
    if bbox_s is None:
        return
    x1, y1, x2, y2 = map(int, bbox_s)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame_bgr.shape[1] - 1, x2)
    y2 = min(frame_bgr.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    txt = f"{name}  {sim:.2f}" if name else f"UNKNOWN  {sim:.2f}"
    cv2.putText(frame_bgr, txt, (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--infer_interval", type=float, default=0.05, help="0.05 ~20Hz")
    ap.add_argument("--window", default="ESP32CAM Face")
    args = ap.parse_args()

    engine = FaceEngine(
        db_host="localhost",
        db_port=5432,
        dbname="espface_db",
        db_user="espface",
        db_password="espraspi",
        model_name="buffalo_l",
        det_size=(640, 640),
        ctx_id=-1,
        face_accept_thresh=0.60,
        face_margin=0.05,
        det_min_score=0.35,
    )

    # estado overlay
    st_lock = threading.Lock()
    face_ts = 0.0
    face_name: Optional[str] = None
    face_sim = 0.0
    bbox_s: Optional[Tuple[float, float, float, float]] = None

    # frame latest
    frame_lock = threading.Lock()
    last_frame: Optional[np.ndarray] = None
    last_frame_ts = 0.0
    frame_seq = 0

    def on_face(res: FaceResult):
        nonlocal face_ts, face_name, face_sim, bbox_s
        now = time.time()
        with st_lock:
            face_ts = now
            face_name = res.name if res.is_known else None
            face_sim = float(res.sim)
            if res.bbox is not None:
                x1, y1, x2, y2 = res.bbox
                if bbox_s is None:
                    bbox_s = (float(x1), float(y1), float(x2), float(y2))
                else:
                    a = BBOX_EMA_ALPHA
                    sx1, sy1, sx2, sy2 = bbox_s
                    bbox_s = (
                        a * float(x1) + (1 - a) * sx1,
                        a * float(y1) + (1 - a) * sy1,
                        a * float(x2) + (1 - a) * sx2,
                        a * float(y2) + (1 - a) * sy2,
                    )

    alg = CAlgFace(cam_id=0, engine=engine, on_face=on_face, min_interval=args.infer_interval)
    alg.start()

    stop_evt = threading.Event()

    async def ws_handler(websocket):
        nonlocal last_frame, last_frame_ts, frame_seq
        print("ESP32-CAM connected:", websocket.remote_address)
        try:
            async for message in websocket:
                if isinstance(message, str):
                    continue
                frame = jpeg_bytes_to_bgr(message)
                if frame is None:
                    continue

                with frame_lock:
                    last_frame = frame
                    last_frame_ts = time.time()

                alg.push_frame(frame, seq=frame_seq)
                frame_seq += 1

        except websockets.exceptions.ConnectionClosed:
            print("ESP32-CAM disconnected")
        except Exception as e:
            print("WS handler error:", e)

    async def ws_server_task():
        async with websockets.serve(
            ws_handler,
            args.host,
            args.port,
            max_size=20 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        ):
            print(f"WebSocket server listening on ws://{args.host}:{args.port}")
            while not stop_evt.is_set():
                await asyncio.sleep(0.1)

    def ui_loop():
        while not stop_evt.is_set():
            with frame_lock:
                frame = None if last_frame is None else last_frame.copy()

            if frame is None:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(canvas, "Waiting for ESP32-CAM...", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"ws://{args.host}:{args.port}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
                cv2.imshow(args.window, canvas)
                if (cv2.waitKey(30) & 0xFF) == 27:
                    stop_evt.set()
                continue

            now = time.time()
            with st_lock:
                alive = (face_ts > 0) and ((now - face_ts) <= FACE_HOLD_SEC)
                nm = face_name
                sm = face_sim
                bb = bbox_s if alive else None

            if bb is not None:
                draw_overlay_inplace(frame, nm, sm, bb)

            cv2.imshow(args.window, frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                stop_evt.set()

        cv2.destroyAllWindows()

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=lambda: loop.run_until_complete(ws_server_task()), daemon=True)
    t.start()

    try:
        ui_loop()
    finally:
        stop_evt.set()
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
        alg.stop()
        engine.close()


if __name__ == "__main__":
    main()
