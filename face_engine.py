"""
face_engine.py
--------------
แกนกลางของระบบ: ดาวน์โหลดโมเดล, ตรวจจับใบหน้า (YuNet), สร้าง embedding และ
จดจำใบหน้า (SFace) พร้อมจัดการฐานข้อมูลใบหน้าที่ลงทะเบียนไว้

ใช้เฉพาะ opencv-python หลัก (มี FaceDetectorYN / FaceRecognizerSF ในตัว)
ทำงานได้บน Debian 13 และดิสโทร Linux อื่น ๆ บน CPU ได้อย่างมีประสิทธิภาพ
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# ที่อยู่ไฟล์/โมเดล
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "faces_db.npz"

# โมเดลจาก OpenCV Zoo (ขนาดเล็ก โหลดครั้งเดียว)
YUNET_FILE = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_FILE = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

# เกณฑ์ความเหมือน (cosine) — มากกว่านี้ถือว่าเป็นคนเดียวกัน
COSINE_THRESHOLD = 0.363  # ค่ามาตรฐานของ SFace


def _download(url: str, dst: Path) -> None:
    """ดาวน์โหลดไฟล์โมเดลถ้ายังไม่มี"""
    if dst.exists() and dst.stat().st_size > 0:
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[models] กำลังดาวน์โหลด {dst.name} ...")
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(dst)
    print(f"[models] เสร็จสิ้น: {dst.name}")


def ensure_models() -> None:
    """ตรวจสอบ/ดาวน์โหลดโมเดลทั้งสองให้พร้อมใช้งาน"""
    _download(YUNET_URL, YUNET_FILE)
    _download(SFACE_URL, SFACE_FILE)


class FaceEngine:
    """ห่อหุ้มตัวตรวจจับ + ตัวจดจำใบหน้า และฐานข้อมูลใบหน้า"""

    def __init__(self, score_threshold: float = 0.8, nms_threshold: float = 0.3):
        ensure_models()

        # YuNet: ตัวตรวจจับใบหน้า (กำหนด input size ทีหลังด้วย setInputSize)
        self.detector = cv2.FaceDetectorYN.create(
            model=str(YUNET_FILE),
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=5000,
        )

        # SFace: ตัวสร้าง embedding 128 มิติ สำหรับเปรียบเทียบใบหน้า
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=str(SFACE_FILE),
            config="",
        )

        # ฐานข้อมูล: ชื่อ -> embedding เฉลี่ย (normalize แล้ว)
        self.names: list[str] = []
        self.embeddings: np.ndarray = np.empty((0, 128), dtype=np.float32)
        self.load_db()

    # ------------------------------------------------------------------ detect
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """คืนค่า array ของใบหน้า (n, 15): x,y,w,h, 5 จุดสำคัญ, score"""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    # --------------------------------------------------------------- embedding
    def embed(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """จัดแนวใบหน้าแล้วสร้าง embedding ที่ normalize เป็นเวกเตอร์หนึ่งหน่วย"""
        aligned = self.recognizer.alignCrop(frame, face_row)
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    # --------------------------------------------------------------- recognize
    def recognize(self, embedding: np.ndarray) -> tuple[str, float]:
        """เทียบ embedding กับฐานข้อมูล คืน (ชื่อ, คะแนนความเหมือน)"""
        if len(self.names) == 0:
            return "Unknown", 0.0
        # ทุกเวกเตอร์ normalize แล้ว -> dot product = cosine similarity
        sims = self.embeddings @ embedding
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= COSINE_THRESHOLD:
            return self.names[idx], score
        return "Unknown", score

    # ----------------------------------------------------------------- database
    def load_db(self) -> None:
        if DB_PATH.exists():
            data = np.load(DB_PATH, allow_pickle=True)
            self.names = list(data["names"])
            self.embeddings = data["embeddings"].astype(np.float32)
            print(f"[db] โหลดใบหน้าที่ลงทะเบียน {len(self.names)} คน")

    def save_db(self) -> None:
        np.savez(
            DB_PATH,
            names=np.array(self.names, dtype=object),
            embeddings=self.embeddings,
        )
        print(f"[db] บันทึกแล้ว ({len(self.names)} คน) -> {DB_PATH}")

    def add_person(self, name: str, embeddings: list[np.ndarray]) -> None:
        """เพิ่มคนใหม่ด้วยค่าเฉลี่ยของหลาย embedding (แล้ว normalize อีกครั้ง)"""
        mean = np.mean(np.stack(embeddings), axis=0)
        mean /= np.linalg.norm(mean) + 1e-9
        if name in self.names:  # อัปเดตของเดิม
            i = self.names.index(name)
            self.embeddings[i] = mean
        else:
            self.names.append(name)
            self.embeddings = np.vstack([self.embeddings, mean[None, :]])
        self.save_db()
