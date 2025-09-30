import os
import torch
import numpy as np
import cv2
from lprnet.lprnet import build_lprnet
from lprnet.utils import predict_image
from inference import get_model
import supervision as sv

# --------------------
# CONFIG
# --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
API_KEY = "Dq8TDPGHIXHOjFfdoQHp"   # replace with your real key
os.environ["ROBOFLOW_API_KEY"] = API_KEY

# --------------------
# LOAD LPRNet
# --------------------
lpr_model = build_lprnet().to(DEVICE)
lpr_model.load_state_dict(torch.load("chinese_lprnet_best.pth", map_location=DEVICE))
lpr_model.eval()

# --------------------
# LOAD YOLO DETECTOR
# --------------------
detector = get_model(model_id="chinese-license-plate-detection/1")

# --------------------
# READ IMAGE
# --------------------
image_file = "images/image.png"
image = cv2.imread(image_file)

# --------------------
# RUN DETECTION
# --------------------
results = detector.infer(image)[0]
detections = sv.Detections.from_inference(results)

# annotate for visualization
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
sv.plot_image(annotated_image)

# --------------------
# PROCESS EACH BOUNDING BOX
# --------------------
os.makedirs("outputs", exist_ok=True)

for i, (x_min, y_min, x_max, y_max) in enumerate(detections.xyxy):
    # crop the bounding box
    cropped = image[int(y_min):int(y_max), int(x_min):int(x_max)]

    # convert BGR -> RGB
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    # resize to (width=94, height=24)
    resized = cv2.resize(cropped_rgb, (94, 24))

    # convert to (3,H,W)q
    chw = np.transpose(resized, (2, 0, 1))  # (3,24,94)

    # convert to tensor
    chw_tensor = torch.from_numpy(chw).float().to(DEVICE) / 255.0

    # predict with LPRNet
    prediction = predict_image(lpr_model, chw_tensor)
    print(f"Prediction for plate {i}: {prediction}")

    # save cropped plate (still in HWC for cv2.imwrite)
    cv2.imwrite(f"outputs/plate_{i}_{prediction}.png",
                cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

    print(f"Saved outputs/plate_{i}_{prediction}.png with CHW shape {chw.shape}")
