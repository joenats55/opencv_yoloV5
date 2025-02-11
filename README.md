# การตรวจจับวัตถุด้วย YOLOv5 และ OpenCV

## ภาพรวม
โครงการนี้ใช้ **YOLOv5** ในการตรวจจับวัตถุผ่านกล้องที่เชื่อมต่อ โดยระบบจะรับภาพวิดีโอจากกล้อง ประมวลผลด้วย YOLOv5 และแสดงผลวัตถุที่ตรวจจับได้ในรูปแบบของกรอบสี่เหลี่ยม ผู้ใช้สามารถโต้ตอบกับระบบโดยคลิกบนวิดีโอเพื่อเลือกจุดที่สนใจ

## คุณสมบัติ
- ใช้ **YOLOv5** สำหรับการตรวจจับวัตถุแบบเรียลไทม์
- อ่านเฟรมวิดีโอจากกล้อง (`/dev/video2`)
- แสดงวัตถุที่ตรวจจับได้พร้อมกรอบสี่เหลี่ยม
- รองรับการโต้ตอบของผู้ใช้ผ่านการคลิกเมาส์เพื่อเลือกจุดที่สนใจ
- ออกจากโปรแกรมได้โดยกดปุ่ม `q`

## ความต้องการของระบบ
- Python 3.x
- OpenCV (`cv2`)
- PyTorch
- โมเดล YOLOv5 (`best.pt`)
- NumPy

1. ติดตั้งไลบรารีที่จำเป็น:
   ```sh
   pip install numpy opencv-python torch torchvision torchaudio pandas
   ```
2. ดาวน์โหลดโมเดล YOLOv5 ของพวกนายเอง (`best.pt`) และวางไว้ในโฟลเดอร์ของโปรเจกต์

## วิธีการใช้งาน
1. รันสคริปต์โดยใช้คำสั่ง:
   ```sh
   python fliename.py
   ```
2. ระบบทำหน้าที่:
   - เปิดกล้องเพื่อรับภาพวิดีโอ
   - ใช้ YOLOv5 ตรวจจับวัตถุในแต่ละเฟรม
   - แสดงผลภาพที่ผ่านการประมวลผล พร้อมกรอบสี่เหลี่ยมรอบวัตถุที่ตรวจจับได้
   - อนุญาตให้ผู้ใช้คลิกที่ภาพเพื่อเลือกจุดสนใจ
3. กด `q` เพื่อออกจากโปรแกรม

## โครงสร้างของโค้ด
- **โหลดโมเดล:** ใช้ `torch.hub.load()` ในการโหลดโมเดล YOLOv5
- **อ่านภาพจากกล้อง:** ใช้ OpenCV ในการรับเฟรมจากกล้อง
- **การประมวลผลภาพ:** โมเดลจะตรวจจับวัตถุในแต่ละเฟรม
- **การแสดงผล:** ระบบจะวาดกรอบรอบวัตถุที่พบ และแสดงพิกัดของจุดที่เลือก
- **การออกจากโปรแกรม:** กด `q` เพื่อปิดระบบ



