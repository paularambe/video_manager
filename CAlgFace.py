#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Callable, List, Tuple

import numpy as np
from insightface.app import FaceAnalysis

from faceDatabase import FaceDatabase, PersonEmbedding


# ============================
# Resultado
# ============================

@dataclass
class FaceResult:
    cam_id: int
    ts: float
    has_face: bool
    is_known: bool
    name: Optional[str]
    sim: float
    bbox: Optional[Tuple[int, int, int, int]]
    det_score: float
    frame_seq: Optional[int] = None


# ============================
# FaceEngine (DB + cache shared, FaceAnalysis thread-local)
# ============================

class FaceEngine:
    """
    Compartido por TODO el proceso para DB + cache de centroides,
    pero con FaceAnalysis thread-local para evitar serialización entre cámaras.
    """

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5432,
        dbname: str = "espface_db",
        db_user: str = "espface",
        db_password: str = "espraspi",
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (512, 512),   # más rápido que 640
        ctx_id: int = -1,                         # CPU
        face_accept_thresh: float = 0.60,
        face_margin: float = 0.05,
        det_min_score: float = 0.35,
        providers: Optional[List[str]] = None,
    ):
        self.face_accept_thresh = float(face_accept_thresh)
        self.face_margin = float(face_margin)
        self.det_min_score = float(det_min_score)

        self.model_name = model_name
        self.det_size = tuple(map(int, det_size))
        self.ctx_id = int(ctx_id)
        self.providers = providers or ["CPUExecutionProvider"]

        self._cache_lock = threading.Lock()
        self._tls = threading.local()  # FaceAnalysis por hilo

        self.db = FaceDatabase(
            host=db_host,
            port=db_port,
            dbname=dbname,
            user=db_user,
            password=db_password,
        )

        self._centroids: Dict[str, np.ndarray] = {}
        self.reload_cache()

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass

    def reload_cache(self):
        embs = self.db.load_all_embeddings()
        centroids = self._build_centroids(embs)
        with self._cache_lock:
            self._centroids = centroids

    @staticmethod
    def _build_centroids(embs: List[PersonEmbedding]) -> Dict[str, np.ndarray]:
        by_name: Dict[str, List[np.ndarray]] = {}
        for pe in embs:
            by_name.setdefault(pe.name, []).append(pe.embedding)

        centroids: Dict[str, np.ndarray] = {}
        for name, vecs in by_name.items():
            m = np.mean(np.stack(vecs, axis=0), axis=0)
            n = float(np.linalg.norm(m))
            if n > 0:
                m = m / n
            centroids[name] = m
        return centroids

    def _get_face_app(self) -> FaceAnalysis:
        """
        Crea FaceAnalysis una vez por hilo.
        """
        app = getattr(self._tls, "face_app", None)
        if app is None:
            app = FaceAnalysis(name=self.model_name, providers=self.providers)
            app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
            self._tls.face_app = app
        return app

    def detect_faces(self, frame_bgr):
        app = self._get_face_app()
        faces = app.get(frame_bgr)
        if not faces:
            return []
        out = []
        for f in faces:
            sc = float(getattr(f, "det_score", 0.0))
            if sc >= self.det_min_score:
                out.append(f)
        return out

    def match_embedding(self, emb_normed: np.ndarray) -> Tuple[Optional[str], float, float]:
        with self._cache_lock:
            centroids = dict(self._centroids)

        if not centroids:
            return None, 0.0, 0.0

        best_name = None
        best_sim = -1.0
        second_sim = -1.0

        for name, c in centroids.items():
            sim = float(np.dot(emb_normed, c))
            if sim > best_sim:
                second_sim = best_sim
                best_sim = sim
                best_name = name
            elif sim > second_sim:
                second_sim = sim

        if best_sim >= self.face_accept_thresh and (best_sim - second_sim) >= self.face_margin:
            return best_name, best_sim, second_sim

        return None, best_sim, second_sim

    def enroll_embeddings_bulk(self, name: str, embs_normed: List[np.ndarray]):
        if not embs_normed:
            return
        person_id = self.db.get_or_create_person(name)
        for emb in embs_normed:
            self.db.add_embedding(person_id, emb)
        self.reload_cache()


# ============================
# CAlgFace (por cámara / hilo)
# ============================

class CAlgFace:
    """
    1 hilo por cámara:
    - procesa SOLO el último frame (drop)
    - emit_on_no_face para mantener overlay estable
    - debug timing: dt_pred, infer, push2pred
    """

    def __init__(
        self,
        cam_id: int,
        engine: FaceEngine,
        on_face: Callable[[FaceResult], None],
        min_interval: float = 0.0,           # 0 => a tope
        pick_largest_face: bool = True,
        emit_on_no_face: bool = True,
        no_face_cooldown: float = 0.12,

        debug_timing: bool = False,
        timing_print_every: int = 10,

        wait_timeout: float = 0.003,         # más realtime (ojo CPU)
    ):
        self.cam_id = int(cam_id)
        self.engine = engine
        self.on_face = on_face

        self.min_interval = float(min_interval)
        self.pick_largest_face = bool(pick_largest_face)

        self.emit_on_no_face = bool(emit_on_no_face)
        self.no_face_cooldown = float(no_face_cooldown)

        self.debug_timing = bool(debug_timing)
        self.timing_print_every = int(timing_print_every)
        self.wait_timeout = float(wait_timeout)

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._frame_lock = threading.Lock()
        self._new_frame_evt = threading.Event()
        self._latest_frame = None
        self._latest_seq: Optional[int] = None
        self._latest_push_ts: float = 0.0

        self._last_proc_seq: Optional[int] = None
        self._last_infer_ts: float = 0.0
        self._last_no_face_emit_ts: float = 0.0

        self._pred_count: int = 0
        self._last_pred_ts: float = 0.0

    def start(self):
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self._new_frame_evt.set()
        if self._thread is not None:
            try:
                self._thread.join(timeout=0.7)
            except Exception:
                pass
        self._thread = None

    def push_frame(self, frame_bgr, seq: Optional[int] = None):
        if frame_bgr is None:
            return
        now = time.time()
        with self._frame_lock:
            self._latest_frame = frame_bgr
            self._latest_seq = seq
            self._latest_push_ts = now
        self._new_frame_evt.set()

    def predict_once(self, frame_bgr, frame_seq: Optional[int] = None) -> FaceResult:
        ts = time.time()
        try:
            faces = self.engine.detect_faces(frame_bgr)
        except Exception:
            faces = []

        if not faces:
            return FaceResult(
                cam_id=self.cam_id, ts=ts,
                has_face=False, is_known=False, name=None, sim=0.0,
                bbox=None, det_score=0.0, frame_seq=frame_seq
            )

        face = faces[0]
        if self.pick_largest_face and len(faces) > 1:
            face = sorted(
                faces,
                key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
                reverse=True
            )[0]

        x1, y1, x2, y2 = face.bbox
        bbox = (int(x1), int(y1), int(x2), int(y2))
        det_score = float(getattr(face, "det_score", 0.0))

        emb = face.normed_embedding
        name, best_sim, _second = self.engine.match_embedding(emb)

        return FaceResult(
            cam_id=self.cam_id,
            ts=ts,
            has_face=True,
            is_known=(name is not None),
            name=name,
            sim=float(best_sim),
            bbox=bbox,
            det_score=det_score,
            frame_seq=frame_seq,
        )

    def _emit(self, res: FaceResult):
        try:
            self.on_face(res)
        except Exception:
            pass

    def _loop(self):
        while not self._stop_evt.is_set():
            self._new_frame_evt.wait(timeout=self.wait_timeout)
            self._new_frame_evt.clear()
            if self._stop_evt.is_set():
                break

            if self.min_interval > 0:
                now = time.time()
                dt = now - self._last_infer_ts
                if dt < self.min_interval:
                    time.sleep(max(0.0, self.min_interval - dt))

            with self._frame_lock:
                frame = self._latest_frame
                seq = self._latest_seq
                push_ts = self._latest_push_ts

            if frame is None:
                continue

            if seq is not None and self._last_proc_seq is not None and seq == self._last_proc_seq:
                continue

            t0 = time.perf_counter()
            now_wall = time.time()
            prev_pred_ts = self._last_pred_ts
            push2pred_ms = (now_wall - push_ts) * 1000.0 if push_ts > 0 else 0.0

            res = self.predict_once(frame, frame_seq=seq)

            t1 = time.perf_counter()
            infer_ms = (t1 - t0) * 1000.0

            self._last_infer_ts = time.time()
            self._last_proc_seq = seq

            self._pred_count += 1
            dt_pred_ms = 0.0 if prev_pred_ts <= 0 else (now_wall - prev_pred_ts) * 1000.0
            self._last_pred_ts = now_wall

            if self.debug_timing and (self._pred_count % self.timing_print_every == 0):
                print(
                    f"[CAlgFace cam={self.cam_id}] "
                    f"dt_pred={dt_pred_ms:6.1f}ms | infer={infer_ms:6.1f}ms | push2pred={push2pred_ms:6.1f}ms | seq={seq}"
                )

            if res.has_face:
                self._emit(res)
            else:
                if self.emit_on_no_face:
                    now = time.time()
                    if (now - self._last_no_face_emit_ts) >= self.no_face_cooldown:
                        self._last_no_face_emit_ts = now
                        self._emit(res)
