import cv2
import time
from ultralytics import YOLO

# -------- CONFIGURATION --------
MODEL_PATH = r"C:\Users\singh.neeraj\OneDrive - DALMIA BHARAT LIMITED\Desktop\office\Dalmia_KPI\best.pt"  # Change to your model path

# Replace with the actual RTSP URL from your camera
RTSP_URL = "rtsp://admin:abc123!@@192.168.191.40:554/Streaming/channels/101"

CONFIDENCE = 0.60

print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded successfully.")

print("[INFO] Connecting to RTSP stream...")
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[ERROR] Cannot open RTSP stream.")
    print("Check the RTSP URL, credentials, port and camera settings.")
    raise SystemExit(1)

print("[INFO] RTSP connected. Starting inference...")

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to read a frame from RTSP.")
            break

        # Run inference on each live frame
        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            verbose=False
        )

        # Draw segmentation masks without bounding boxes
        output = results[0].plot(boxes=False)
        frame_count += 1

        # Display inference FPS
        elapsed = max(time.time() - start_time, 0.001)
        fps = frame_count / elapsed

        cv2.putText(
            output,
            f"Inference FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("RTSP Live Model Test", output)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Inference stopped.")