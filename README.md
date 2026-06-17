# Face Scan + Vitals HUD

ระบบสแกน ตรวจจับ และจดจำใบหน้าด้วย OpenCV แบบเรียลไทม์ พร้อมแสดงสถานะ
อุณหภูมิร่างกายและอัตราการเต้นของหัวใจ (ค่าเทส/placeholder)

ใช้โมเดล **YuNet** (ตรวจจับ) + **SFace** (จดจำ) ซึ่งมากับ `opencv-python` หลัก
ไม่ต้องใช้ dlib หรือ opencv-contrib — ทำงานเร็วบน CPU บน Debian 13 และดิสโทรอื่น

## เลย์เอาต์ HUD
- **มุมขวาบน**  → ชื่อผู้ใช้ที่จดจำได้
- **มุมซ้ายล่าง** → อุณหภูมิร่างกาย + อัตราการเต้นของหัวใจ (ค่าเทส)

## ติดตั้ง

### Debian 13 / Ubuntu
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv libgl1 v4l-utils
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Fedora / RHEL
```bash
sudo dnf install -y python3 python3-pip mesa-libGL
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Arch
```bash
sudo pacman -S --needed python python-pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> โมเดล ONNX (YuNet, SFace) จะถูกดาวน์โหลดอัตโนมัติครั้งแรกที่รัน ลงในโฟลเดอร์ `models/`
> หากเครื่องไม่มีอินเทอร์เน็ต ให้ดาวน์โหลดล่วงหน้าจาก OpenCV Zoo แล้ววางใน `models/`

## วิธีใช้

### 1) ลงทะเบียนใบหน้า (ทำครั้งเดียวต่อคน)
```bash
python3 enroll.py --name "ชื่อของคุณ"
```
มองกล้องแล้วกด **SPACE** เก็บใบหน้าหลาย ๆ มุม (ค่าเริ่มต้น 15 ครั้ง) ระบบบันทึกอัตโนมัติ

### 2) รันแอปหลัก
```bash
python3 app.py
```
กด **q** เพื่อออก

## โครงสร้างไฟล์
| ไฟล์ | หน้าที่ |
|------|---------|
| `face_engine.py` | ดาวน์โหลดโมเดล, ตรวจจับ/สร้าง embedding/จดจำ, จัดการฐานข้อมูล |
| `enroll.py` | ลงทะเบียนใบหน้าผู้ใช้จากเว็บแคม |
| `app.py` | แอปหลัก แสดงผล HUD แบบเรียลไทม์ |
| `faces_db.npz` | ฐานข้อมูลใบหน้า (สร้างหลังลงทะเบียน) |

## ต่อยอด
ค่าอุณหภูมิและอัตราการเต้นของหัวใจในตอนนี้เป็น **ค่าเทส** กำหนดไว้ใน `app.py`
(`TEST_TEMPERATURE`, `TEST_HEART_RATE`) ภายหลังสามารถแทนที่ด้วยข้อมูลจากเซนเซอร์จริง
(เช่น MLX90614 สำหรับอุณหภูมิ, MAX30102 สำหรับ heart rate) ได้ทันที
# cv-pun
