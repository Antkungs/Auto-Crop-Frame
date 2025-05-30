from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8x.pt")

target_class = 39

os.makedirs("output/images", exist_ok=True)
os.makedirs("output/labels", exist_ok=True)

video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
object_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame, conf=0.8, classes=[target_class])

    boxes = results[0].boxes.xyxy
    classes = results[0].boxes.cls.tolist()

    if len(boxes) > 0:
        height, width, _ = frame.shape

        for i, (box, class_id) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = map(int, box)
            cropped = frame[y1:y2, x1:x2]

            object_count += 1
            img_name = f"frame{frame_count:04d}_obj{object_count:02d}.jpg"
            label_name = img_name.replace(".jpg", ".txt")

            cv2.imwrite(f"output/images/{img_name}", cropped)

            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            with open(f"output/labels/{label_name}", "w") as f:
                f.write(f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

cap.release()
print(f"✔ Done: Saved {object_count} cropped objects from {frame_count} frames.")
