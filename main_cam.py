import cv2
import torch
import numpy as np
from ultralytics import YOLO

# -------------------------
# Load YOLOv11 detection model
# -------------------------
yolo_model = YOLO("yolo_detector.pt")

# -------------------------
# Load your OCR model (LPRNet or similar)
# -------------------------
from lprnet.lprnet import build_lprnet
from lprnet.utils import predict_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ocr_model = build_lprnet()  # adjust class_num
ocr_model.load_state_dict(torch.load("5000_chinese_lprnet_best_0_05484444214209147.pth", map_location=DEVICE))
ocr_model.to(DEVICE).eval()


# -------------------------
# Preprocess cropped plate
# -------------------------
def preprocess_plate(crop):
    crop = cv2.resize(crop, (94, 24))          # resize to W=94, H=24
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = torch.from_numpy(crop).float() / 255.0
    crop = crop.permute(2, 0, 1)               # HWC -> CHW
    return crop

# -------------------------
# Run realtime detection + OCR
# -------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect license plates
    results = yolo_model(frame, conf=0.5)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])   # bounding box
        crop = frame[y1:y2, x1:x2]               # crop plate

        if crop.size != 0:  # ensure valid crop
            plate_tensor = preprocess_plate(crop)
            text = predict_image(ocr_model, plate_tensor)

            # Draw bounding box + text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("License Plate Detection + OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
