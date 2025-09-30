import os
import cv2
import torch
import shutil
from glob import glob

# -------------------------
# Load OCR model
# -------------------------
from lprnet.lprnet import build_lprnet
from lprnet.utils import predict_image, greedy_decode, idx_to_char, blank_idx

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ocr_model = build_lprnet(class_num=len(idx_to_char))
ocr_model.load_state_dict(torch.load("chinese_lprnet_best.pth", map_location=DEVICE))
ocr_model.to(DEVICE).eval()

# -------------------------
# Preprocess image
# -------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (94, 24))  # (W,H)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1)  # HWC -> CHW
    return img

# -------------------------
# Directories
# -------------------------
input_folder = "crops"
labeled_folder = "labeled"
duplicate_folder = "duplicates"
os.makedirs(labeled_folder, exist_ok=True)
os.makedirs(duplicate_folder, exist_ok=True)

# -------------------------
# Labeling
# -------------------------
seen_names = set()

for img_path in glob(os.path.join(input_folder, "*.*")):
    plate_tensor = preprocess_image(img_path)
    if plate_tensor is None:
        continue

    # Predict text
    text = predict_image(ocr_model, plate_tensor)

    # Output file paths
    base_name = f"{text}.jpg"
    save_path = os.path.join(labeled_folder, base_name)

    if not os.path.exists(save_path):
        # First time → save to main labeled folder
        shutil.copy(img_path, save_path)
    else:
        # Duplicate → save to duplicates folder with suffix
        suffix = 1
        while True:
            dup_name = f"{text}_{suffix}.jpg"
            dup_path = os.path.join(duplicate_folder, dup_name)
            if not os.path.exists(dup_path):
                shutil.copy(img_path, dup_path)
                break
            suffix += 1

print("✅ Labeling completed!")
