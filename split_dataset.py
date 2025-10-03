import os
import shutil
import random

# -------------------------
# CONFIG
# -------------------------
INPUT_DIR = "./sampled_ccpd_all_cropped"   # folder gốc chứa ảnh
OUTPUT_DIR = "./lrpnet_dataset_5500"         # folder output
VAL_SIZE = 0.1                   # tỉ lệ validation (test sẽ cùng tỉ lệ)
SEED = 42                        # random seed để reproducible

random.seed(SEED)

# -------------------------
# Chuẩn bị output folder
# -------------------------
for split in ["train", "val", "test"]:
    split_path = os.path.join(OUTPUT_DIR, split)
    os.makedirs(split_path, exist_ok=True)

# -------------------------
# Lấy danh sách file ảnh
# -------------------------
files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(files)

n_total = len(files)
n_val = int(n_total * VAL_SIZE)
n_test = n_val
n_train = n_total - n_val - n_test

print(f"Tổng số ảnh: {n_total}")
print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

# -------------------------
# Chia ảnh
# -------------------------
train_files = files[:n_train]
val_files = files[n_train:n_train+n_val]
test_files = files[n_train+n_val:]

splits = {"train": train_files, "val": val_files, "test": test_files}

for split, split_files in splits.items():
    for f in split_files:
        src = os.path.join(INPUT_DIR, f)
        dst = os.path.join(OUTPUT_DIR, split, f)
        shutil.copy2(src, dst)

print("Hoàn tất split dataset!")
