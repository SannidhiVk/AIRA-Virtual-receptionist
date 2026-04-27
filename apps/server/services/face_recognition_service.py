"""
face_recognition_service.py
----------------------------
Verifies an employee's identity by comparing a live camera frame
(captured from the frontend) against the photo stored in the database.

HOW IT WORKS:
1. The frontend captures ONE JPEG frame when the employee says their name.
2. That frame is sent as base64 over WebSocket with the spoken name.
3. This service:
   a) Looks up the employee by name in the DB → gets their photo_path
   b) Runs DeepFace.verify(stored_photo, live_frame) using the Facenet512 model
   c) Saves the live frame to a captures folder for cross-verification
   d) Returns a structured result: { verified, distance, message }

TROUBLESHOOTING distance ≈ 1.0 (guaranteed mismatch):
- Distance of 1.082 (>1.0) means face was NOT detected in one or both images.
  DeepFace then compares raw pixel embeddings → always huge distance.
- Fix: ensure the stored photo and live frame both contain a clearly visible,
  front-facing face. Check the saved captures in receptionist/photos/captures/.
- Use FACE_VERIFY_DETECTOR env var to try 'retinaface' or 'mtcnn' for better
  detection (slower but much more robust than default 'opencv').
"""

import base64
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

# --- ADD THESE TWO LINES ---
from dotenv import load_dotenv

load_dotenv()
# ---------------------------

logger = logging.getLogger(__name__)

# ── Threshold tuning ──────────────────────────────────────────────────────────
VERIFY_THRESHOLD = float(os.getenv("FACE_VERIFY_THRESHOLD", "0.68"))

# Change the default fallback from "yolov8" to "mtcnn" or "opencv"
DETECTOR_BACKEND = os.getenv("FACE_VERIFY_DETECTOR", "mtcnn")

# Model name
MODEL_NAME = os.getenv("FACE_VERIFY_MODEL", "ArcFace")

# Photo storage root — relative to this file's location (apps/server/)
PHOTOS_DIR = (
    Path(__file__).resolve().parent.parent / "receptionist" / "photos" / "employees"
)

# Captures directory — live frames are saved here for cross-verification
CAPTURES_DIR = (
    Path(__file__).resolve().parent.parent / "receptionist" / "photos" / "captures"
)


def _ensure_photos_dir() -> None:
    """Create the photos directory and captures directory if they don't exist yet."""
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)


def get_photo_path(employee_id: int) -> Path:
    """Return the expected disk path for an employee's stored photo."""
    return PHOTOS_DIR / f"{employee_id}.jpg"


def warmup_deepface():
    """
    Runs a dummy verification in the background on startup.
    This forces TensorFlow and the ArcFace/MTCNN models to load into memory
    so the first user doesn't experience a 40-second delay.
    """
    logger.info("Starting DeepFace warmup sequence in the background...")
    try:
        from deepface import DeepFace

        # Create a dummy blank 224x224 image
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

        # Run a fake verification with enforce_detection=False so it doesn't crash on the blank image
        DeepFace.verify(
            img1_path=dummy_img,
            img2_path=dummy_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
        logger.info("✅ DeepFace warmup complete! Live requests will now be fast.")
    except Exception as e:
        logger.warning(f"DeepFace warmup failed: {e}")


def _save_capture(
    image_b64: str, employee_name: str, verified: bool, distance: float
) -> Optional[Path]:
    """
    Save the live capture to CAPTURES_DIR for cross-verification.
    Filename format: <timestamp>_<employee>_<result>_d<distance>.jpg
    Returns the saved path, or None on failure.
    """
    try:
        # Strip data URL prefix if present
        raw_b64 = image_b64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]

        image_bytes = base64.b64decode(raw_b64)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = employee_name.replace(" ", "_")[:30]
        result_tag = "MATCH" if verified else "MISMATCH"
        dist_tag = f"d{distance:.3f}".replace(".", "p")
        filename = f"{ts}_{safe_name}_{result_tag}_{dist_tag}.jpg"
        save_path = CAPTURES_DIR / filename
        save_path.write_bytes(image_bytes)
        logger.info("Capture saved to: %s", save_path)
        return save_path
    except Exception as e:
        logger.warning("Could not save capture for '%s': %s", employee_name, e)
        return None


def decode_b64_to_tempfile(image_b64: str) -> Optional[str]:
    """
    Decode a base64 image string (from the browser canvas) into a
    temporary file on disk. DeepFace needs a file path or numpy array.

    Returns the temp file path, or None on failure.
    The caller is responsible for deleting the temp file after use.
    """
    try:
        # Strip the data URL prefix if present: "data:image/jpeg;base64,..."
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_b64)

        # Write to a named temp file that DeepFace can read
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(image_bytes)
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception as e:
        logger.error("Failed to decode base64 image: %s", e)
        return None


def verify_employee_face(audio_name: str, image_b64: str) -> dict:
    """
    Main verification function called from the WebSocket handler.

    Args:
        audio_name:  The employee name spoken by the person (from Whisper/LLM).
        image_b64:   Base64-encoded JPEG of the live camera frame.

    Returns a dict:
        {
            "verified":    bool   — True if face matches stored photo
            "distance":    float  — DeepFace distance (lower = more similar)
            "message":     str    — Human-readable result for Jarvis TTS
            "has_photo":   bool   — False if no photo is stored yet (skip check)
            "employee_id": int|None
        }
    """
    _ensure_photos_dir()

    # ── Step 1: Look up the employee from DB ──────────────────────────────────
    employee = _get_employee_by_name(audio_name)

    if not employee:
        logger.warning("Face verify: employee '%s' not found in DB.", audio_name)
        return {
            "verified": True,  # Can't verify → don't block the flow
            "distance": -1.0,
            "message": "",
            "has_photo": False,
            "employee_id": None,
        }

    # ── Step 2: Check if a stored photo exists ────────────────────────────────
    stored_photo_path = get_photo_path(employee.id)

    if not employee.photo_path or not stored_photo_path.exists():
        logger.info(
            "Face verify: employee '%s' has no stored photo — skipping verification.",
            audio_name,
        )
        return {
            "verified": True,  # No photo to compare → don't block
            "distance": -1.0,
            "message": "",
            "has_photo": False,
            "employee_id": employee.id,
        }

    # ── Step 3: Write the live frame to a temp file ───────────────────────────
    tmp_path = decode_b64_to_tempfile(image_b64)
    if not tmp_path:
        return {
            "verified": True,  # Decode failure → don't block
            "distance": -1.0,
            "message": "",
            "has_photo": True,
            "employee_id": employee.id,
        }

    # ── Step 4: Run DeepFace comparison ───────────────────────────────────────
    try:
        from deepface import DeepFace  # Lazy import — only loaded when first used

        logger.info(
            "Running DeepFace.verify | model=%s | detector=%s | stored=%s | live=%s",
            MODEL_NAME,
            DETECTOR_BACKEND,
            stored_photo_path,
            tmp_path,
        )

        result = DeepFace.verify(
            img1_path=str(stored_photo_path),  # Employee's stored photo from DB
            img2_path=tmp_path,  # Live capture from webcam
            model_name=MODEL_NAME,  # Facenet512 — higher accuracy than Facenet128
            detector_backend=DETECTOR_BACKEND,  # Configurable via env var
            enforce_detection=True,  # Don't crash if face isn't perfectly detected
            distance_metric="cosine",  # Works well with Facenet family
            align=True,  # Face alignment greatly improves accuracy
        )

        verified: bool = result.get("verified", False)
        distance: float = result.get("distance", 1.0)

        # Log raw DeepFace threshold vs our custom threshold
        # Log raw DeepFace threshold vs our custom threshold
        deepface_threshold = result.get("threshold", VERIFY_THRESHOLD)
        logger.info(
            "DeepFace raw result | distance=%.4f | deepface_threshold=%.4f | our_threshold=%.2f",
            distance,
            deepface_threshold,
            VERIFY_THRESHOLD,
        )

        # Distance > 0.9 almost certainly means face was NOT detected.
        if distance > 0.9:
            logger.warning(
                "Distance %.4f is suspiciously high (>0.9) for '%s' — "
                "face likely NOT detected in stored photo or live frame. "
                "Check stored photo at: %s  |  Check live capture in CAPTURES_DIR.",
                distance,
                audio_name,
                stored_photo_path,
            )

        # Override with our own threshold for extra control
        if distance > VERIFY_THRESHOLD:
            verified = False

        logger.info(
            "Face verify for '%s': verified=%s, distance=%.4f (threshold=%.2f)",
            audio_name,
            verified,
            distance,
            VERIFY_THRESHOLD,
        )

        if verified:
            message = (
                f"Identity verified for {audio_name}. "
                f"You can proceed with your question or request."
            )
        else:
            message = (
                f"I can see someone in the camera, but it doesn't quite match "
                f"the photo we have on file for {audio_name}. "
                f"Could you confirm your identity?"
            )

        # ── Step 5: Save the live capture for cross-verification ─────────────
        _save_capture(image_b64, audio_name, verified, distance)

        return {
            "verified": verified,
            "distance": round(distance, 4),
            "message": message,
            "has_photo": True,
            "employee_id": employee.id,
            "capture_dir": str(CAPTURES_DIR),
        }

    except Exception as e:
        logger.error(
            "DeepFace verification failed for '%s': %s", audio_name, e, exc_info=True
        )
        # Save the raw capture even on error so you can inspect it
        _save_capture(image_b64, audio_name, False, -1.0)

        # --- UPDATE THIS RETURN BLOCK ---
        # Do NOT return verified: True on a crash. Return False and a message.
        return {
            "verified": False,
            "distance": -1.0,
            "message": "I'm sorry, I encountered an error while trying to verify your face. Please try again.",
            "has_photo": True,
            "employee_id": employee.id,
        }

    finally:
        # Always clean up the temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _get_employee_by_name(name: str):
    """
    Fetch an Employee ORM object by name.
    Uses the existing difflib-based fuzzy lookup from the database layer.
    Returns None if not found.
    """
    try:
        from receptionist.database import get_employee_by_name

        return get_employee_by_name(name)
    except Exception as e:
        logger.error("DB lookup for employee '%s' failed: %s", name, e)
        return None
