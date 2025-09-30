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

# annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# --------------------
# CAMERA LOOP
# --------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run detection
    results = detector.infer(frame)[0]
    detections = sv.Detections.from_inference(results)

    labels = []
    for (x_min, y_min, x_max, y_max) in detections.xyxy:
        # crop bbox
        cropped = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        if cropped.size == 0:
            continue

        # BGR -> RGB
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # resize to (width=94, height=24)
        resized = cv2.resize(cropped_rgb, (94, 24))

        # HWC -> CHW
        chw = np.transpose(resized, (2, 0, 1))  # (3,24,94)

        # tensor
        chw_tensor = torch.from_numpy(chw).float().to(DEVICE) / 255.0

        # predict
        prediction = predict_image(lpr_model, chw_tensor)

        labels.append(prediction)  # show only the characters

    # annotate frame with boxes + predictions
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    if labels:
        annotated_frame = label_annotator.annotate(scene=annotated_frame,
                                                   detections=detections,
                                                   labels=labels)

    # show live video
    cv2.imshow("License Plate Recognition", annotated_frame)

    # quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
