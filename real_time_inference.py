from ultralytics import YOLO
import cv2
import time

# Load trained model
model = YOLO("runs/detect/augmented_model/weights/best.pt")

# Class names
class_names = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole"
]

# Video source
video_path = "road_video.mp4"   # change to camera index 0 for webcam

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video")
    exit()

prev_time = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO inference
    results = model(frame, conf=0.25)

    # Draw detections
    for r in results:

        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = class_names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            text = f"{label} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,0,255), 2)

    cv2.imshow("Road Damage Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()