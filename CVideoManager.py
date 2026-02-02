#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import asyncio
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import websockets

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QSizePolicy,
    QGridLayout,
    QScrollArea,
)

from CAlgFace import FaceEngine, CAlgFace, FaceResult

# (Opcional) Gesture; si no existe, no rompe.
try:
    from CAlgGesture import CAlgGesture
except Exception:
    CAlgGesture = None


RECOGNIZE_GESTURES = 0
RECOGNIZE_FACES = 1

THUMB_W, THUMB_H = 320, 240
GRID_COLS = 3

GUI_REFRESH_MS = 33
STREAM_DROP_SEC = 300.0

FACE_HOLD_SEC = 0.60
BBOX_EMA_ALPHA = 0.65

# Más realtime
ALG_FACE_MIN_INTERVAL = 0.0       # 0 => a tope (si CPU se dispara, pon 0.01-0.02)
ALG_FACE_WAIT_TIMEOUT = 0.003     # más reactivo (si CPU se dispara, pon 0.005-0.01)
ALG_FACE_DEBUG_TIMING = True
ALG_FACE_TIMING_EVERY = 10


@dataclass
class CameraWidgets:
    box: QGroupBox
    title_label: QLabel
    led: QLabel
    fps_label: QLabel
    video_label: QLabel
    gesture_label: QLabel


@dataclass
class CameraContext:
    cam_id: int
    name: str

    latest_frame_bgr: Optional[np.ndarray]
    frame_lock: threading.Lock

    last_rx_time: float
    fps_rx: float
    _fps_last_ts: float

    detect_lock: threading.Lock

    alg_face: Optional[CAlgFace]
    alg_gesture: Optional[Any]

    frame_seq: int

    face_ts: float
    face_bbox: Optional[Tuple[int, int, int, int]]
    face_bbox_s: Optional[Tuple[float, float, float, float]]
    face_name: Optional[str]
    face_sim: float
    face_det: float

    last_result: Dict[str, Any]
    widgets: Optional[CameraWidgets]


class CVideoManager(QMainWindow):
    sig_cam_added = pyqtSignal(int)
    sig_cam_removed = pyqtSignal(int)

    def __init__(self, ws_host="0.0.0.0", ws_port=8765, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ESP32-CAM Video Manager")

        self.ws_host = ws_host
        self.ws_port = ws_port

        self.grid_cols = GRID_COLS
        self.grid_next_index = 0

        self.ws_thread = None
        self.ws_server_started = False

        self.cams: Dict[int, CameraContext] = {}
        self.cams_lock = threading.Lock()
        self.cam_id_counter = 0

        self.face_seen_lock = threading.Lock()
        self.last_seen_face_text = ""
        self.last_seen_face_ts = 0.0

        self.face_engine: Optional[FaceEngine] = None
        if RECOGNIZE_FACES:
            self.face_engine = FaceEngine(
                db_host="localhost",
                db_port=5432,
                dbname="espface_db",
                db_user="espface",
                db_password="espraspi",
                model_name="buffalo_l",
                det_size=(512, 512),
                ctx_id=-1,
                face_accept_thresh=0.60,
                face_margin=0.05,
                det_min_score=0.35,
            )
            print("[INFO] FaceEngine listo (FaceAnalysis por hilo).")

        self._build_ui()

        # Estos métodos EXISTEN en este archivo (ya no te falla)
        self.sig_cam_added.connect(self._on_cam_added_gui)
        self.sig_cam_removed.connect(self._on_cam_removed_gui)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_all_cameras)
        self.timer.start(GUI_REFRESH_MS)

        self._start_ws_server()

    def closeEvent(self, event):
        with self.cams_lock:
            ids = list(self.cams.keys())
        for cam_id in ids:
            self.sig_cam_removed.emit(cam_id)

        if self.face_engine is not None:
            self.face_engine.close()

        super().closeEvent(event)

    # ---------------- UI ----------------

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        video_group = QGroupBox("Cámaras")
        root.addWidget(video_group)

        self.face_global_label = QLabel("Caras: --")
        self.face_global_label.setStyleSheet("color: #00cc66; font-weight: 600;")
        self.face_global_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root.addWidget(self.face_global_label)

        group_layout = QVBoxLayout(video_group)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        group_layout.addWidget(self.scroll)

        self.grid_container = QWidget()
        self.scroll.setWidget(self.grid_container)

        self.cameras_grid = QGridLayout(self.grid_container)
        self.cameras_grid.setContentsMargins(10, 10, 10, 10)
        self.cameras_grid.setHorizontalSpacing(10)
        self.cameras_grid.setVerticalSpacing(10)

        self.no_cameras_label = QLabel("Sin cámaras conectadas.\nEnciende tus ESP32-CAM y se añadirán aquí.")
        self.no_cameras_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_cameras_label.setStyleSheet("color: #aaaaaa;")
        self.cameras_grid.addWidget(self.no_cameras_label, 0, 0, 1, GRID_COLS)

    def _grid_position_for_next(self) -> tuple[int, int]:
        idx = self.grid_next_index
        row = idx // self.grid_cols
        col = idx % self.grid_cols
        self.grid_next_index += 1
        return row, col

    def _create_camera_widgets(self, ctx: CameraContext):
        if self.no_cameras_label is not None:
            self.no_cameras_label.hide()

        box = QGroupBox()
        box.setTitle("")
        box.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        outer = QVBoxLayout(box)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        header = QWidget()
        h = QHBoxLayout(header)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        title_label = QLabel(ctx.name)
        title_label.setStyleSheet("font-weight: 600;")

        led = QLabel()
        led.setFixedSize(12, 12)
        led.setStyleSheet("background-color: #aa0000; border-radius: 6px;")

        fps_label = QLabel("0.0 FPS")
        fps_label.setStyleSheet("color: #dddddd;")
        fps_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        h.addWidget(title_label)
        h.addWidget(led)
        h.addStretch(1)
        h.addWidget(fps_label)
        outer.addWidget(header)

        video_holder = QWidget()
        video_holder.setFixedSize(THUMB_W, THUMB_H)

        video_label = QLabel("Esperando vídeo…", video_holder)
        video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_label.setFixedSize(THUMB_W, THUMB_H)
        video_label.setStyleSheet("background-color: #111111; border: 1px solid #333333;")
        outer.addWidget(video_holder)

        gesture_label = QLabel("Gesto: --")
        gesture_label.setStyleSheet("color: #cccccc;")
        gesture_label.setWordWrap(True)
        outer.addWidget(gesture_label)

        row, col = self._grid_position_for_next()
        self.cameras_grid.addWidget(box, row, col)

        ctx.widgets = CameraWidgets(
            box=box,
            title_label=title_label,
            led=led,
            fps_label=fps_label,
            video_label=video_label,
            gesture_label=gesture_label,
        )

    def _reflow_grid(self):
        with self.cams_lock:
            cams = sorted(self.cams.values(), key=lambda c: c.cam_id)
        self.grid_next_index = 0

        while self.cameras_grid.count():
            item = self.cameras_grid.takeAt(0)
            w = item.widget()
            if w is not None and w is not self.no_cameras_label:
                self.cameras_grid.removeWidget(w)

        if not cams:
            self.no_cameras_label.show()
            self.cameras_grid.addWidget(self.no_cameras_label, 0, 0, 1, self.grid_cols)
            return

        self.no_cameras_label.hide()

        for ctx in cams:
            if ctx.widgets is None:
                continue
            row, col = self._grid_position_for_next()
            self.cameras_grid.addWidget(ctx.widgets.box, row, col)

    # ---------------- WS ----------------

    def _start_ws_server(self):
        if self.ws_server_started:
            return
        self.ws_server_started = True

        def run_ws():
            async def handler(websocket):
                addr = websocket.remote_address
                cam_id = self._register_camera()
                print(f"[WS] Cliente conectado: {addr} -> cam_id={cam_id}")

                try:
                    async for message in websocket:
                        if isinstance(message, bytes):
                            np_arr = np.frombuffer(message, np.uint8)
                            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            now = time.time()

                            with self.cams_lock:
                                ctx = self.cams.get(cam_id)

                            if ctx is not None and frame_bgr is not None:
                                with ctx.frame_lock:
                                    ctx.latest_frame_bgr = frame_bgr

                                if ctx.alg_face is not None:
                                    ctx.alg_face.push_frame(frame_bgr, seq=ctx.frame_seq)
                                    ctx.frame_seq += 1

                                dt = now - ctx._fps_last_ts
                                if dt > 0:
                                    inst = 1.0 / dt
                                    ctx.fps_rx = (0.85 * ctx.fps_rx) + (0.15 * inst) if ctx.fps_rx > 0 else inst
                                ctx._fps_last_ts = now
                                ctx.last_rx_time = now

                except Exception as e:
                    print(f"[WS] Error con cliente {addr} (cam_id={cam_id}): {e}")
                finally:
                    print(f"[WS] Cliente desconectado: {addr} (cam_id={cam_id})")
                    self.sig_cam_removed.emit(cam_id)

            async def main_ws():
                print(f"[WS] Iniciando servidor en {self.ws_host}:{self.ws_port} ...")
                async with websockets.serve(handler, self.ws_host, self.ws_port, max_size=2**23):
                    print("[WS] Servidor listo. Esperando ESP32-CAM...")
                    await asyncio.Future()

            asyncio.run(main_ws())

        self.ws_thread = threading.Thread(target=run_ws, daemon=True)
        self.ws_thread.start()

    # ---------------- Alta/Baja cámara ----------------

    def _register_camera(self) -> int:
        with self.cams_lock:
            cam_id = self.cam_id_counter
            self.cam_id_counter += 1

            now = time.time()
            ctx = CameraContext(
                cam_id=cam_id,
                name=f"Cámara {cam_id + 1}",
                latest_frame_bgr=None,
                frame_lock=threading.Lock(),
                last_rx_time=0.0,
                fps_rx=0.0,
                _fps_last_ts=now,
                detect_lock=threading.Lock(),
                alg_face=None,
                alg_gesture=None,
                frame_seq=0,

                face_ts=0.0,
                face_bbox=None,
                face_bbox_s=None,
                face_name=None,
                face_sim=0.0,
                face_det=0.0,

                last_result={"label_text": "Gesto: --", "label_color": "#cccccc"},
                widgets=None,
            )
            self.cams[cam_id] = ctx

        # crea UI en hilo GUI
        self.sig_cam_added.emit(cam_id)
        # arranca algoritmos
        self._start_algorithms_for_camera(cam_id)
        return cam_id

    def _start_algorithms_for_camera(self, cam_id: int):
        with self.cams_lock:
            ctx = self.cams.get(cam_id)
        if ctx is None:
            return

        def publish_face(res: FaceResult):
            now = time.time()
            with ctx.detect_lock:
                ctx.face_ts = now
                ctx.face_bbox = res.bbox
                ctx.face_name = res.name if res.is_known else None
                ctx.face_sim = float(res.sim)
                ctx.face_det = float(res.det_score)

                if res.bbox is not None:
                    x1, y1, x2, y2 = res.bbox
                    if ctx.face_bbox_s is None:
                        ctx.face_bbox_s = (float(x1), float(y1), float(x2), float(y2))
                    else:
                        a = BBOX_EMA_ALPHA
                        sx1, sy1, sx2, sy2 = ctx.face_bbox_s
                        ctx.face_bbox_s = (
                            a * float(x1) + (1 - a) * sx1,
                            a * float(y1) + (1 - a) * sy1,
                            a * float(x2) + (1 - a) * sx2,
                            a * float(y2) + (1 - a) * sy2,
                        )

            if res.has_face:
                msg = f"{res.name} visto en {ctx.name}! ({res.sim:.2f})" if (res.is_known and res.name) \
                    else f"Cara detectada en {ctx.name} (desconocida, {res.sim:.2f})"
                with self.face_seen_lock:
                    self.last_seen_face_text = msg
                    self.last_seen_face_ts = now

        if RECOGNIZE_FACES and self.face_engine is not None:
            ctx.alg_face = CAlgFace(
                cam_id=cam_id,
                engine=self.face_engine,
                on_face=publish_face,
                min_interval=ALG_FACE_MIN_INTERVAL,
                pick_largest_face=True,
                emit_on_no_face=True,
                no_face_cooldown=0.12,
                debug_timing=ALG_FACE_DEBUG_TIMING,
                timing_print_every=ALG_FACE_TIMING_EVERY,
                wait_timeout=ALG_FACE_WAIT_TIMEOUT,
            )
            ctx.alg_face.start()

    def _remove_camera(self, cam_id: int):
        with self.cams_lock:
            ctx = self.cams.pop(cam_id, None)
        if ctx is None:
            return

        if ctx.alg_face is not None:
            ctx.alg_face.stop()

        if ctx.alg_gesture is not None and hasattr(ctx.alg_gesture, "stop"):
            try:
                ctx.alg_gesture.stop()
            except Exception:
                pass

        if ctx.widgets is not None:
            self.cameras_grid.removeWidget(ctx.widgets.box)
            ctx.widgets.box.setParent(None)
            ctx.widgets.box.deleteLater()

        self._reflow_grid()

        with self.cams_lock:
            if not self.cams:
                self.no_cameras_label.show()
                self.grid_next_index = 0

    # --------- ESTOS SON LOS QUE TE FALTABAN ----------
    def _on_cam_added_gui(self, cam_id: int):
        with self.cams_lock:
            ctx = self.cams.get(cam_id)
        if ctx is None:
            return
        if ctx.widgets is None:
            self._create_camera_widgets(ctx)
        else:
            self._reflow_grid()

    def _on_cam_removed_gui(self, cam_id: int):
        self._remove_camera(cam_id)
    # ---------------------------------------------------

    # ---------------- GUI refresh ----------------

    def _draw_face_overlay_inplace(self, frame_bgr: np.ndarray, ctx: CameraContext, now: float):
        with ctx.detect_lock:
            ts = ctx.face_ts
            name = ctx.face_name
            sim = ctx.face_sim
            bbox_s = ctx.face_bbox_s

        if bbox_s is None or ts <= 0.0 or (now - ts) > FACE_HOLD_SEC:
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

    def process_all_cameras(self):
        now = time.time()
        with self.cams_lock:
            cam_list = list(self.cams.values())

        # etiqueta global
        with self.face_seen_lock:
            txt = self.last_seen_face_text
            ts = self.last_seen_face_ts
        if txt and (now - ts) < 2.0:
            self.face_global_label.setText(txt)
        else:
            self.face_global_label.setText("Caras: --")

        for ctx in cam_list:
            if ctx.widgets is None:
                continue

            # LED / FPS (opcional)
            age = (now - ctx.last_rx_time) if ctx.last_rx_time > 0 else 999.0
            if ctx.last_rx_time > 0 and age < 1.0:
                ctx.widgets.led.show()
            else:
                ctx.widgets.led.hide()

            ctx.widgets.fps_label.setText(f"{ctx.fps_rx:4.1f} FPS")

            with ctx.frame_lock:
                frame = None if ctx.latest_frame_bgr is None else ctx.latest_frame_bgr.copy()

            if frame is None:
                continue

            self._draw_face_overlay_inplace(frame, ctx, now)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            hh, ww, ch = rgb.shape
            qimg = QImage(rgb.data, ww, hh, ch * ww, QImage.Format.Format_RGB888).copy()

            pixmap = QPixmap.fromImage(qimg).scaled(
                ctx.widgets.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            ctx.widgets.video_label.setPixmap(pixmap)


def main():
    app = QApplication(sys.argv)
    win = CVideoManager(ws_host="0.0.0.0", ws_port=8765)
    win.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
