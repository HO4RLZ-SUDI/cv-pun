"""
enroll.py
---------
ลงทะเบียนใบหน้าผู้ใช้จากกล้องเว็บแคม

วิธีใช้:
    python3 enroll.py --name "ชื่อผู้ใช้"
    python3 enroll.py --name "Alice" --samples 20 --camera 0

กด SPACE เพื่อเก็บตัวอย่างใบหน้า, กด q เพื่อยกเลิก
เมื่อเก็บครบตามจำนวน --samples ระบบจะบันทึกอัตโนมัติ
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from face_engine import FaceEngine


def main() -> int:
    ap = argparse.ArgumentParser(description="ลงทะเบียนใบหน้าผู้ใช้")
    ap.add_argument("--name", required=True, help="ชื่อผู้ใช้ที่จะลงทะเบียน")
    ap.add_argument("--samples", type=int, default=15, help="จำนวนตัวอย่างที่ต้องเก็บ")
    ap.add_argument("--camera", type=int, default=1, help="หมายเลขกล้อง (ค่าเริ่มต้น 0)")
    args = ap.parse_args()

    engine = FaceEngine()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[error] เปิดกล้องหมายเลข {args.camera} ไม่ได้", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    collected = []
    win = "Enroll - auto capture  (SPACE=manual  q=quit)"
    print(f"[enroll] กำลังลงทะเบียน '{args.name}' — เก็บอัตโนมัติเมื่อเจอใบหน้า (หรือกด SPACE เอง)")

    last_auto = 0.0          # เวลาเก็บอัตโนมัติครั้งล่าสุด
    AUTO_INTERVAL = 0.4      # เก็บทุก ~0.4 วินาทีเมื่อมีใบหน้า

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[error] อ่านภาพจากกล้องไม่ได้", file=sys.stderr)
            break

        faces = engine.detect(frame)
        # เลือกใบหน้าที่ใหญ่ที่สุด (ใกล้กล้องสุด)
        best = None
        if len(faces) > 0:
            best = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = best[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # สถานะการตรวจจับ ให้ผู้ใช้รู้ว่าเจอหน้าหรือยัง
        status = "FACE OK" if best is not None else "NO FACE - move closer / add light"
        status_color = (0, 255, 0) if best is not None else (40, 40, 220)
        cv2.putText(frame, f"{args.name}: {len(collected)}/{args.samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[enroll] ยกเลิก")
            break

        # เก็บตัวอย่าง: อัตโนมัติเมื่อมีใบหน้า หรือกด SPACE เอง
        now = time.time()
        manual = key == ord(" ")
        auto = best is not None and (now - last_auto) > AUTO_INTERVAL
        if best is not None and (manual or auto):
            collected.append(engine.embed(frame, best))
            last_auto = now
            print(f"[enroll] เก็บตัวอย่าง {len(collected)}/{args.samples}")

        if len(collected) >= args.samples:
            engine.add_person(args.name, collected)
            print(f"[enroll] ลงทะเบียน '{args.name}' สำเร็จ")
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
