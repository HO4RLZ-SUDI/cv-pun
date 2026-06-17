"""
app.py
------
แอปหลัก: สแกน/ตรวจจับ/จดจำใบหน้าแบบเรียลไทม์ พร้อมแสดง HUD

เลย์เอาต์ HUD:
  - มุมขวาบน  : ชื่อผู้ใช้ที่จดจำได้
  - มุมซ้ายล่าง: อุณหภูมิร่างกาย + อัตราการเต้นของหัวใจ (เป็นค่าเทส/placeholder)

วิธีใช้:
    python3 app.py
    python3 app.py --camera 0

กด q เพื่อออก
"""

from __future__ import annotations

import argparse
import random
import time

import cv2

from face_engine import FaceEngine

# ---------------------------------------------------------------------------
# ค่าเทส (placeholder) ของสถานะร่างกาย — ภายหลังค่อยต่อกับเซนเซอร์จริง
# ---------------------------------------------------------------------------
TEST_TEMPERATURE = 36.6   # องศาเซลเซียส
TEST_HEART_RATE = 72      # ครั้ง/นาที (bpm)

# สี (BGR)
GREEN = (0, 220, 0)
RED = (40, 40, 220)
WHITE = (255, 255, 255)
CYAN = (255, 220, 0)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _wobble(base: float, amount: float) -> float:
    """ทำให้ค่าเทสขยับเล็กน้อยเหมือนค่าจริง เพื่อความสมจริงของการแสดงผล"""
    return base + random.uniform(-amount, amount)


def draw_label(img, text, org, color, scale=0.7, thick=2):
    """วาดข้อความพร้อมพื้นหลังโปร่งให้อ่านง่ายบนทุกฉากหลัง"""
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + bl + 2), BLACK, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return tw, th


def main() -> int:
    ap = argparse.ArgumentParser(description="สแกน/จดจำใบหน้า + HUD สถานะร่างกาย")
    ap.add_argument("--camera", type=int, default=1, help="หมายเลขกล้อง (ค่าเริ่มต้น 1)")
    args = ap.parse_args()

    engine = FaceEngine()
    if len(engine.names) == 0:
        print("[warn] ยังไม่มีใบหน้าที่ลงทะเบียน — รัน enroll.py ก่อนเพื่อให้จดจำชื่อได้")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[error] เปิดกล้องหมายเลข {args.camera} ไม่ได้")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    win = "Face Scan + Vitals (q=quit)"
    prev_t = time.time()
    fps = 0.0
    # อัปเดตค่าเทสทุก ~1 วินาที เพื่อไม่ให้ตัวเลขกระพริบเร็วเกินไป
    last_vital_t = 0.0
    temp_disp, hr_disp = TEST_TEMPERATURE, TEST_HEART_RATE

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[error] อ่านภาพจากกล้องไม่ได้")
            break

        H, W = frame.shape[:2]
        faces = engine.detect(frame)

        # ชื่อผู้ใช้ของใบหน้าที่เด่นที่สุด (ใหญ่สุด) ไว้โชว์มุมขวาบน
        primary_name = "No face"

        # เรียงจากใหญ่ -> เล็ก เพื่อให้ใบหน้าหลักคือตัวแรก
        order = sorted(range(len(faces)), key=lambda i: -faces[i][2] * faces[i][3])
        for rank, i in enumerate(order):
            face = faces[i]
            x, y, w, h = face[:4].astype(int)
            emb = engine.embed(frame, face)
            name, score = engine.recognize(emb)

            color = GREEN if name != "Unknown" else RED
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            tag = f"{name} {score:.2f}" if name != "Unknown" else "Unknown"
            draw_label(frame, tag, (x, max(y - 10, 20)), color, scale=0.6, thick=2)

            if rank == 0:
                primary_name = name

        # ---- อัปเดตค่าเทสของสถานะร่างกาย ----
        now = time.time()
        if now - last_vital_t > 1.0:
            temp_disp = _wobble(TEST_TEMPERATURE, 0.2)
            hr_disp = int(_wobble(TEST_HEART_RATE, 3))
            last_vital_t = now

        # ---- FPS ----
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # ===================================================================
        # HUD
        # ===================================================================
        # มุมขวาบน: ชื่อผู้ใช้
        name_text = primary_name
        (tw, th), _ = cv2.getTextSize(name_text, FONT, 0.9, 2)
        draw_label(frame, name_text, (W - tw - 16, 34), CYAN, scale=0.9, thick=2)

        # มุมซ้ายล่าง: อุณหภูมิ + อัตราการเต้นของหัวใจ (ค่าเทส)
        temp_text = f"Temp: {temp_disp:.1f} C"
        hr_text = f"HR:   {hr_disp} bpm"
        draw_label(frame, hr_text, (12, H - 16), WHITE, scale=0.7, thick=2)
        draw_label(frame, temp_text, (12, H - 48), WHITE, scale=0.7, thick=2)
        draw_label(frame, "[TEST DATA]", (12, H - 80), (0, 180, 255), scale=0.5, thick=1)

        # FPS เล็ก ๆ มุมซ้ายบน
        cv2.putText(frame, f"FPS: {fps:4.1f}", (12, 22),
                    FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow(win, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
