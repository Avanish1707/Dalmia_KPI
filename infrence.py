import cv2
import numpy as np
import time
from ultralytics import YOLO

# -------- CONFIGURATION --------
MODEL_PATH = r"C:\Users\singh.neeraj\OneDrive - DALMIA BHARAT LIMITED\Desktop\office\Dalmia_KPI\best.pt"  # Change to your model path

# Replace with the actual RTSP URL from your camera
RTSP_URL = "rtsp://admin:abc123!@@192.168.191.40:554/Streaming/channels/101"

CONFIDENCE = 0.45
CHAIR_CONFIDENCE = 0.70

print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded successfully.")
print(f"[INFO] Model classes: {model.names}")

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
            imgsz=1240,
            verbose=False
        )

        # Fill small gaps in each mask and draw masks without bounding boxes.
        result = results[0]
        output = frame.copy()
        chair_states = {}

        if result.masks is not None and result.boxes is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            masks = result.masks.data.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for mask, class_id, confidence in zip(masks, class_ids, confidences):
                class_name = str(model.names[class_id]).strip()
                normalized_name = class_name.lower()
                has_person = "person" in normalized_name
                has_chair = "chair" in normalized_name
                has_without = "without" in normalized_name
                is_person_with_chair = has_person and has_chair and not has_without

                if is_person_with_chair and confidence < CHAIR_CONFIDENCE:
                    continue

                if is_person_with_chair:
                    chair_states[class_name] = max(
                        chair_states.get(class_name, 0.0), float(confidence)
                    )

                mask = cv2.resize(
                    mask,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                binary_mask = cv2.morphologyEx(
                    binary_mask, cv2.MORPH_CLOSE, kernel
                )
                mask_area = binary_mask > 0

                if not np.any(mask_area):
                    continue

                color = (
                    (255, 160, 0)
                    if has_without
                    else (0, 200, 0) if is_person_with_chair else (255, 160, 0)
                )
                output[mask_area] = (
                    output[mask_area].astype(np.float32) * 0.55
                    + np.asarray(color, dtype=np.float32) * 0.45
                ).astype(np.uint8)

                ys, xs = np.where(mask_area)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(
                    output,
                    label,
                    (int(xs.min()), max(int(ys.min()) - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        status_text = (
            " | ".join(
                f"{class_name} ({confidence:.2f})"
                for class_name, confidence in chair_states.items()
            )
            if chair_states
            else "Chair status: Not detected"
        )
        cv2.putText(
            output,
            status_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        frame_count += 1

        # Display inference FPS
        elapsed = max(time.time() - start_time, 0.001)
        fps = frame_count / elapsed

        cv2.putText(
            output,
            f"Inference FPS: {fps:.1f}",
            (20, 70),
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