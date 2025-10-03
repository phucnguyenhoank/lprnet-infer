import os
import cv2
import torch
from ultralytics import YOLO

# -------------------------
# Load YOLO detection model
# -------------------------
yolo_model = YOLO("yolo_detector.pt")

# -------------------------
# INPUT / OUTPUT CONFIG
# -------------------------
INPUT_DIR = "./sampled_ccpd"     # folder chứa ảnh gốc
OUTPUT_DIR = "./sampled_ccpd_cropped"      # folder lưu ảnh cắt biển số

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_label(filename: str) -> str:
    """
    Chuyển '川A_FQ263.jpg' -> 'AFQ263.jpg'
    """
    name, ext = os.path.splitext(filename)
    # Tách phần sau dấu gạch dưới (nếu có)
    if "_" in name:
        parts = name.split("_")
        # Lấy phần 1 và 2, bỏ ký tự Chinese
        if len(parts) >= 2:
            new_label = parts[0][1:] + parts[1]   # '川A' -> 'A', + 'FQ263'
        else:
            new_label = name
    else:
        new_label = name
    return new_label + ext


def process_image(img_path, output_dir):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Lỗi đọc {img_path}")
        return

    # Detect license plate
    results = yolo_model(img, conf=0.5)

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        if crop.size != 0:
            # Tạo tên file mới từ nhãn
            fname = os.path.basename(img_path)
            new_name = clean_label(fname)

            # Nếu nhiều plate trong 1 ảnh -> thêm số thứ tự
            if i > 0:
                name, ext = os.path.splitext(new_name)
                new_name = f"{name}_{i}{ext}"

            save_path = os.path.join(output_dir, new_name)
            cv2.imwrite(save_path, crop)
            print(f"Đã lưu {save_path}")


# -------------------------
# Chạy qua toàn bộ folder
# -------------------------
files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
print(f"Tìm thấy {len(files)} ảnh để xử lý")

for f in files:
    process_image(os.path.join(INPUT_DIR, f), OUTPUT_DIR)

print("Hoàn tất!")
