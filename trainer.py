#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
import websockets

from CAlgFace import FaceEngine


def jpeg_bytes_to_bgr(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    if not jpeg_bytes:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def largest_face(faces):
    if not faces:
        return None
    if len(faces) == 1:
        return faces[0]
    return sorted(
        faces,
        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        reverse=True
    )[0]


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Nombre persona (clave DB)")
    ap.add_argument("--host", default="0.0.0.0", help="Host WS server (Raspi)")
    ap.add_argument("--port", type=int, default=8765, help="Puerto WS server")
    ap.add_argument("--samples", type=int, default=100, help="Embeddings a capturar")
    ap.add_argument("--min_face_px", type=int, default=80, help="Cara mínima (px) para aceptar")
    ap.add_argument("--min_blur", type=float, default=60.0, help="Filtro blur (var Laplacian)")
    ap.add_argument("--cooldown_ms", type=int, default=60, help="Tiempo mínimo entre muestras")
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

    # frame compartido (llegado por WS)
    frame_lock = threading.Lock()
    last_frame: Optional[np.ndarray] = None
    last_frame_ts = 0.0

    embs: List[np.ndarray] = []
    capturing = False
    stop_evt = threading.Event()
    last_take_ts = 0.0

    async def ws_handler(websocket):
        nonlocal last_frame, last_frame_ts
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
            print(f"WebSocket trainer listening on ws://{args.host}:{args.port}")
            while not stop_evt.is_set():
                await asyncio.sleep(0.1)

    def ui_loop():
        nonlocal capturing, last_take_ts

        print("\n=== TRAINER ESP32-CAM ===")
        print("Controles:")
        print("  SPACE  -> empezar/parar captura")
        print("  R      -> reset contador")
        print("  Q/ESC  -> salir\n")

        while not stop_evt.is_set():
            with frame_lock:
                frame = None if last_frame is None else last_frame.copy()
                f_age = time.time() - last_frame_ts

            if frame is None:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(canvas, "Waiting for ESP32-CAM...", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"ws://{args.host}:{args.port}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
                cv2.imshow("trainer_esp32cam.py", canvas)
                k = cv2.waitKey(30) & 0xFF
                if k == 27:
                    stop_evt.set()
                continue

            faces = engine.detect_faces(frame)
            face = largest_face(faces)

            status = "NO FACE"
            color = (0, 0, 255)
            can_take = False
            det_score = 0.0
            blur = 0.0

            if face is not None:
                x1, y1, x2, y2 = map(int, face.bbox)
                det_score = float(getattr(face, "det_score", 0.0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                w = x2 - x1
                h = y2 - y1
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    blur = blur_score(gray)

                if w >= args.min_face_px and h >= args.min_face_px and blur >= args.min_blur:
                    can_take = True
                    status = "FACE OK"
                    color = (0, 255, 0)
                else:
                    status = f"FACE (w={w},h={h}) blur={blur:.0f}"
                    color = (0, 165, 255)

                # Captura embeddings si procede
                if capturing and can_take and len(embs) < args.samples:
                    now = time.time()
                    if (now - last_take_ts) * 1000.0 >= args.cooldown_ms:
                        embs.append(face.normed_embedding)
                        last_take_ts = now

            # HUD
            hud1 = f"Name: {args.name}   Capturing: {capturing}   {len(embs)}/{args.samples}"
            hud2 = f"{status}   det={det_score:.2f}   min_px={args.min_face_px}   min_blur={args.min_blur:.0f}"
            cv2.putText(frame, hud1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
            cv2.putText(frame, hud2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.putText(frame, "SPACE=start/stop  R=reset  Q/ESC=quit", (10, frame.shape[0]-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)

            cv2.imshow("trainer_esp32cam.py", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == 32:  # SPACE
                capturing = not capturing
            elif k in (ord('r'), ord('R')):
                embs.clear()
                capturing = False
            elif k in (ord('q'), 27):  # q o ESC
                stop_evt.set()

            # Fin automático
            if len(embs) >= args.samples:
                capturing = False
                stop_evt.set()

        cv2.destroyAllWindows()

    # WS server en hilo asyncio, UI en main thread
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

        # Guardar si hay muestras
        if len(embs) > 0:
            print(f"\nGuardando {len(embs)} embeddings en DB para '{args.name}' ...")
            engine.enroll_embeddings_bulk(args.name, embs)
            print("OK: guardado + cache/centroides recargados (reentreno).")
        else:
            print("\nNo se capturaron embeddings. No guardo nada.")

        engine.close()


if __name__ == "__main__":
    main()
