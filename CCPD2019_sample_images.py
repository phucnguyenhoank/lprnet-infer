import os
import glob
import shutil
import random
import re
from collections import defaultdict

# ==== CONFIG ====
BASE_PATH = "C:/Users/nguye/Downloads/CCPD2019.tar/CCPD2019"  # đường dẫn CCPD2019
OUTPUT_PATH = "./sampled_ccpd"  # nơi lưu ảnh đã chọn
SAMPLE_SIZE = 10000             # số lượng ảnh muốn lấy
MODE = "equal-per-sub"          # "equal-per-sub" hoặc "uniform-random"
RANDOM_SEED = 42
COPY_FILES = True               # True = copy, False = symlink
SUBS = ["ccpd_base", "ccpd_fn", "ccpd_db", "ccpd_rotate", "ccpd_weather", "ccpd_blur"]

# ==== Bảng mã ký tự ====
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
             "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W',
             'X','Y','Z','O']
ads = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X',
       'Y','Z','0','1','2','3','4','5','6','7','8','9','O']

def extract_plate_number_code(plate_token: str) -> str:
    parts = plate_token.split("_")
    if len(parts) < 3:
        return plate_token.replace("_", "")
    chi_let = provinces[int(parts[0])]
    alp_let = alphabets[int(parts[1])]
    alp_num_let = "".join([ads[int(c)] for c in parts[2:]])
    return chi_let + alp_let + " " + alp_num_let

def sanitize_filename(s: str, max_len=200) -> str:
    s = re.sub(r'[<>:"/\\|?*]', '_', s)
    s = s.replace(' ', '_')
    return s[:max_len]

def parse_ccpd_filename(path: str):
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    parts = name.split("-")
    plate_token = parts[4] if len(parts) >= 5 else name
    return plate_token, ext

def main():
    random.seed(RANDOM_SEED)

    # load file list
    files_per_sub = {}
    for sub in SUBS:
        sub_path = os.path.join(BASE_PATH, sub)
        files = sorted(glob.glob(os.path.join(sub_path, "*.*")))
        files_per_sub[sub] = files

    total_available = sum(len(v) for v in files_per_sub.values())
    print("Tổng số ảnh tìm thấy:", total_available)

    sample_size = min(SAMPLE_SIZE, total_available)
    selected_files = []

    if MODE == "uniform-random":
        all_files = [f for fs in files_per_sub.values() for f in fs]
        selected_files = random.sample(all_files, sample_size)
    else:  # equal-per-sub
        nonempty = [s for s in SUBS if len(files_per_sub[s]) > 0]
        per_sub = sample_size // len(nonempty)
        remainder = sample_size % len(nonempty)
        for i, sub in enumerate(nonempty):
            files = files_per_sub[sub]
            quota = per_sub + (1 if i < remainder else 0)
            chosen = random.sample(files, min(quota, len(files)))
            selected_files.extend(chosen)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    name_count = defaultdict(int)
    copied = 0
    for src in selected_files:
        plate_token, ext = parse_ccpd_filename(src)
        plate_text = extract_plate_number_code(plate_token)
        safe_name = sanitize_filename(plate_text)

        dest_name = safe_name + ext
        dest_path = os.path.join(OUTPUT_PATH, dest_name)

        if os.path.exists(dest_path) or name_count[safe_name] > 0:
            name_count[safe_name] += 1
            dest_name = f"{safe_name}_{name_count[safe_name]}{ext}"
            dest_path = os.path.join(OUTPUT_PATH, dest_name)

        try:
            if COPY_FILES:
                shutil.copy2(src, dest_path)
            else:
                os.symlink(os.path.abspath(src), dest_path)
            copied += 1
        except Exception as e:
            print("Lỗi copy:", src, "->", e)

    print(f"Đã xuất {copied}/{sample_size} ảnh vào {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
