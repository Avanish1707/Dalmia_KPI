"""Consent-based known/unknown face recognition with InsightFace Buffalo.

Examples:
  python face_identity_kpi.py --config configs/run_kpi.yaml --kpi face --mode run
  python face_identity_kpi.py --config configs/run_kpi.yaml --kpi face --mode init-db
  python face_identity_kpi.py --config configs/run_kpi.yaml --kpi face --mode enroll --person-code EMP001 --name "Employee name"
  python face_identity_kpi.py --config configs/run_kpi.yaml --kpi face --mode recognize
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import psycopg
import yaml
from insightface.app import FaceAnalysis


def load_config(path: str, kpi: str) -> dict:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)["kpis"][kpi]

    local_path = config_path.with_name("run_kpi_local.yaml")
    if local_path.exists():
        with local_path.open(encoding="utf-8") as local_file:
            local_config = (yaml.safe_load(local_file) or {}).get("kpis", {}).get(kpi, {})
        for key, value in local_config.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    return config


def connect(database: dict):
    required = ("host", "port", "dbname", "user", "password")
    missing = [key for key in required if not database.get(key)]
    if missing:
        raise ValueError(f"Missing database values in config: {', '.join(missing)}")
    return psycopg.connect(**{key: database[key] for key in required})


def make_face_app(config: dict) -> FaceAnalysis:
    providers = config["face_model"].get(
        "providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app = FaceAnalysis(name=config["face_model"].get("name", "buffalo_l"), providers=providers)
    app.prepare(ctx_id=int(config["face_model"].get("ctx_id", -1)), det_size=tuple(config["face_model"].get("det_size", [640, 640])))
    return app


def initialise_database(connection, schema_path: str) -> None:
    with Path(schema_path).open(encoding="utf-8") as schema_file:
        with connection.cursor() as cursor:
            cursor.execute(schema_file.read())
    connection.commit()
    print("[INFO] Face database schema is ready.")


def image_files(directory: Path):
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in extensions)


def encode_image(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Unable to encode enrollment photo")
    return encoded.tobytes()


def enroll(
    config: dict,
    person_code: str,
    display_name: str,
    app: FaceAnalysis | None = None,
) -> None:
    enrollment = config["enrollment"]
    # This is the local folder selected in face_identity.yaml, for example
    # `ashutosh`. Its files are enrolled under the supplied person code.
    photos_dir = Path(enrollment["photos_directory"])
    files = image_files(photos_dir) if photos_dir.exists() else []
    if not files:
        raise FileNotFoundError(f"Put enrollment photos in: {photos_dir}")

    app = app or make_face_app(config)
    min_detection_score = float(enrollment.get("minimum_detection_score", 0.80))
    accepted, rejected = 0, 0
    with connect(config["database"]) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO persons (person_code, display_name, consent_confirmed)
                   VALUES (%s, %s, TRUE)
                   ON CONFLICT (person_code) DO UPDATE SET display_name = EXCLUDED.display_name
                   RETURNING id""",
                (person_code, display_name),
            )
            person_id = cursor.fetchone()[0]

            for path in files:
                image = cv2.imread(str(path))
                faces = app.get(image) if image is not None else []
                if not faces:
                    print(f"[SKIP] No face: {path.name}")
                    rejected += 1
                    continue
                face = max(faces, key=lambda item: item.det_score)
                if face.det_score < min_detection_score:
                    print(f"[SKIP] Low face-detection score ({face.det_score:.2f}): {path.name}")
                    rejected += 1
                    continue

                photo = encode_image(image)
                cursor.execute(
                    """INSERT INTO face_enrollments
                       (person_id, model_name, embedding, embedding_dimension, photo_data, photo_sha256, source_filename)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (photo_sha256) DO NOTHING""",
                    (
                        person_id,
                        config["face_model"].get("name", "buffalo_l"),
                        face.normed_embedding.astype(np.float32).tobytes(),
                        len(face.normed_embedding),
                        photo,
                        hashlib.sha256(photo).hexdigest(),
                        path.name,
                    ),
                )
                accepted += cursor.rowcount
                print(f"[OK] Enrolled {path.name}; face score={face.det_score:.2f}")
        connection.commit()
    print(f"[INFO] Enrollment complete: {accepted} accepted, {rejected} rejected.")


def load_known_embeddings(connection) -> tuple[np.ndarray, list[str]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT p.person_code, e.embedding
               FROM face_enrollments e JOIN persons p ON p.id = e.person_id
               WHERE p.active = TRUE AND p.consent_confirmed = TRUE"""
        )
        rows = cursor.fetchall()
    if not rows:
        raise RuntimeError("No active, consented enrolled faces exist in the database.")
    codes = [row[0] for row in rows]
    vectors = np.vstack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
    return vectors, codes


def recognise(config: dict, app: FaceAnalysis | None = None) -> None:
    app = app or make_face_app(config)
    recognition = config["recognition"]
    threshold = float(recognition["match_threshold"])
    with connect(config["database"]) as connection:
        vectors, person_codes = load_known_embeddings(connection)

    cap = cv2.VideoCapture(config["rtsp_url"], cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Cannot open RTSP stream. Check rtsp_url in the config.")

    print(f"[INFO] Recognition started with {len(person_codes)} enrollment templates. Press Q to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read a frame from RTSP.")
                break
            output = frame.copy()
            for face in app.get(frame):
                query = face.normed_embedding.astype(np.float32)
                scores = vectors @ query  # both vectors are L2-normalised: cosine similarity
                best_index = int(np.argmax(scores))
                similarity = float(scores[best_index])
                is_known = similarity >= threshold
                label = f"Known: {person_codes[best_index]} {similarity:.2f}" if is_known else f"Unknown {similarity:.2f}"
                color = (0, 200, 0) if is_known else (0, 0, 255)
                x1, y1, x2, y2 = (int(value) for value in face.bbox)
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow(config.get("window_name", "Face identity KPI"), output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_face_kpi(config: dict) -> None:
    enrollment = config["enrollment"]
    person_code = enrollment.get("person_code")
    display_name = enrollment.get("display_name")
    if not person_code or not display_name:
        raise ValueError(
            "Set enrollment.person_code and enrollment.display_name in the face config."
        )

    with connect(config["database"]) as connection:
        initialise_database(connection, config["database"]["schema_path"])

    app = make_face_app(config)
    enroll(config, person_code, display_name, app)
    recognise(config, app)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll or recognise consented identities with Buffalo.")
    parser.add_argument("--config", default="configs/run_kpi.yaml")
    parser.add_argument("--kpi", default="face")
    parser.add_argument("--mode", required=True, choices=("run", "init-db", "enroll", "recognize"))
    parser.add_argument("--person-code", help="Required for --mode enroll")
    parser.add_argument("--name", help="Required for --mode enroll")
    args = parser.parse_args()
    config = load_config(args.config, args.kpi)
    if args.mode == "run":
        run_face_kpi(config)
    elif args.mode == "init-db":
        with connect(config["database"]) as connection:
            initialise_database(connection, config["database"]["schema_path"])
    elif args.mode == "enroll":
        if not args.person_code or not args.name:
            parser.error("--person-code and --name are required for --mode enroll")
        enroll(config, args.person_code, args.name)
    else:
        recognise(config)
