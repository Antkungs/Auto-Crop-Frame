# 🎯 Object Cropping & Label Extraction from Video using YOLOv8

โปรเจกต์นี้ใช้ **YOLOv8** ในการตรวจจับวัตถุจากวิดีโอ แล้วตัดภาพวัตถุที่ตรวจจับได้ออกมา พร้อมสร้างไฟล์ label สำหรับงานฝึกโมเดล (YOLO format)

---

## 🚀 Features

- โหลดโมเดล YOLOv8 (`yolov8x.pt`) จาก ultralytics  
- ตรวจจับวัตถุเฉพาะคลาสที่กำหนด (ตัวอย่าง: `target_class = 39`)  
- ตัดภาพ (crop) วัตถุที่ตรวจจับได้ในแต่ละเฟรม  
- บันทึกภาพ cropped ลงในโฟลเดอร์ `output/images`  
- สร้างไฟล์ label `.txt` ในโฟลเดอร์ `output/labels` ตามมาตรฐาน YOLO format  
- กรองผลตรวจจับด้วย confidence threshold ที่ 0.8 เพื่อความแม่นยำ

---

## 🛠 Installation

```bash
pip install ultralytics opencv-python
```
## ⚙ Usage
- เตรียมไฟล์วิดีโอ เช่น video.mp4 ไว้ในโฟลเดอร์โปรเจกต์
- ปรับค่าตัวแปร target_class ในโค้ดให้ตรงกับคลาสที่ต้องการตรวจจับ
- รันสคริปต์ Python
- ดูภาพ cropped และไฟล์ label ที่ถูกบันทึกใน output/images และ output/labels ตามลำดับ

## 📋 YOLO Format Label
- ไฟล์ .txt จะเก็บข้อมูลในรูปแบบ:
```
class_id center_x center_y width height
```
โดยทุกค่าจะเป็นสัดส่วน (normalized) เทียบกับขนาดภาพ (ค่าอยู่ในช่วง 0 ถึง 1)

##💡 Notes
- ค่า target_class เช่น 39 อาจหมายถึง “bottle” หรือวัตถุชนิดอื่น ขึ้นกับ dataset ที่ใช้
- ปรับ confidence threshold ในโค้ดได้ตามต้องการ (ค่าเริ่มต้น 0.8)
- สามารถขยายให้รองรับหลายคลาสพร้อมกันได้
- เหมาะสำหรับเตรียมข้อมูล dataset สำหรับการฝึกโมเดลตรวจจับวัตถุ

## 🔗 References
- Ultralytics YOLOv8 Documentation
- YOLO Annotation Format

