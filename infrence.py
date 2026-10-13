import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Run a YAML-configured KPI detector.")
parser.add_argument("--config", default="configs/run_kpi.yaml")
parser.add_argument("--kpi", default="chair")
args = parser.parse_args()
with Path(args.config).open(encoding="utf-8") as config_file:
    all_config = yaml.safe_load(config_file)

CONFIG = all_config["kpis"][args.kpi]
local_config_path = Path(args.config).with_name("run_kpi_local.yaml")
if local_config_path.exists():
    with local_config_path.open(encoding="utf-8") as config_file:
        local_config = yaml.safe_load(config_file) or {}
    CONFIG.update(local_config.get("kpis", {}).get(args.kpi, {}))

MODEL_PATH = CONFIG["model_path"]
RTSP_URL = CONFIG["rtsp_url"]
CONFIDENCE = float(CONFIG.get("inference", {}).get("confidence", 0.45))
MASK_THRESHOLD = float(CONFIG.get("inference", {}).get("mask_threshold", 0.5))
IMAGE_SIZE = int(CONFIG.get("inference", {}).get("image_size", 640))
CLASS_RULES = CONFIG["classes"]


def normalise_class_name(name):
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def get_rule(class_name):
    name = normalise_class_name(class_name)
    for rule in CLASS_RULES.values():
        if any(normalise_class_name(alias) == name for alias in rule.get("aliases", [])):
            return rule
    return None

print(f"[INFO] Loading YOLO model for KPI: {CONFIG['name']}")
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
            imgsz=IMAGE_SIZE,
            verbose=False
        )

        # Fill small gaps in each mask and draw masks without bounding boxes.
        result = results[0]
        output = frame.copy()
        states = {}

        if result.masks is not None and result.boxes is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            masks = result.masks.data.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for mask, class_id, confidence in zip(masks, class_ids, confidences):
                class_name = str(model.names[class_id]).strip()
                rule = get_rule(class_name)
                if rule is None or confidence < float(rule.get("min_confidence", CONFIDENCE)):
                    continue

                state_name = rule.get("state")
                if state_name:
                    states[state_name] = max(
                        states.get(state_name, 0.0), float(confidence)
                    )

                mask = cv2.resize(
                    mask,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                binary_mask = (mask > MASK_THRESHOLD).astype(np.uint8) * 255
                binary_mask = cv2.morphologyEx(
                    binary_mask, cv2.MORPH_CLOSE, kernel
                )
                mask_area = binary_mask > 0

                if not np.any(mask_area):
                    continue

                color = tuple(int(value) for value in rule.get("color_bgr", [0, 200, 0]))
                output[mask_area] = (
                    output[mask_area].astype(np.float32) * 0.55
                    + np.asarray(color, dtype=np.float32) * 0.45
                ).astype(np.uint8)

                ys, xs = np.where(mask_area)
                label = f"{rule.get('label', class_name)} {confidence:.2f}"
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
                f"{state_name} ({confidence:.2f})"
                for state_name, confidence in states.items()
            )
            if states
            else CONFIG.get("empty_status", "No configured state detected")
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

        cv2.imshow(CONFIG.get("window_name", CONFIG["name"]), output)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Inference stopped.")
