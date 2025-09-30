import cv2
from ultralytics import YOLO

# Load your trained YOLOv11 model
model = YOLO("bestNoAugment.pt")

# Open the webcam (0 = default camera, change to 1,2... if multiple cameras)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLOv11 inference
    results = model(frame, conf=0.5)  # adjust confidence if needed

    # Annotate frame with results
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("License Plate Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
