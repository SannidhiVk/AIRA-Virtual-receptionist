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
   b) Runs DeepFace.verify(stored_photo, live_frame) using the Facenet model
   c) Returns a structured result: { verified, distance, message }

WHY DeepFace + Facenet:
- pip install only — no cmake/dlib build issues on Windows
- Facenet gives ~97-98% accuracy, good for controlled office lighting
- Works fully on CPU (no GPU needed)
- enforce_detection=False = graceful fallback when face is partially out of frame
"""

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Threshold tuning ──────────────────────────────────────────────────────────
# Facenet distance: lower = more similar.
# 0.40 is a good conservative threshold for office environments.
# Raise to 0.50 if you get false mismatches (e.g. glasses, different lighting).
FACENET_THRESHOLD = float(os.getenv("FACE_VERIFY_THRESHOLD", "0.40"))

# Photo storage root — relative to this file's location (apps/server/)
PHOTOS_DIR = (
    Path(__file__).resolve().parent.parent / "receptionist" / "photos" / "employees"
)


def _ensure_photos_dir() -> None:
    """Create the photos directory if it doesn't exist yet."""
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


def get_photo_path(employee_id: int) -> Path:
    """Return the expected disk path for an employee's stored photo."""
    return PHOTOS_DIR / f"{employee_id}.jpg"


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

        result = DeepFace.verify(
            img1_path=str(stored_photo_path),  # Employee's stored photo from DB
            img2_path=tmp_path,  # Live capture from webcam
            model_name="Facenet",  # Best accuracy/speed tradeoff on CPU
            detector_backend="opencv",  # Fastest detector, good enough for frontal faces
            enforce_detection=False,  # Don't crash if face isn't perfectly detected
            distance_metric="cosine",  # Works well with Facenet
        )

        verified: bool = result.get("verified", False)
        distance: float = result.get("distance", 1.0)

        # Also check against our own threshold for extra control
        if distance > FACENET_THRESHOLD:
            verified = False

        logger.info(
            "Face verify for '%s': verified=%s, distance=%.4f (threshold=%.2f)",
            audio_name,
            verified,
            distance,
            FACENET_THRESHOLD,
        )

        if verified:
            message = ""  # No challenge needed — Jarvis continues normally
        else:
            message = (
                f"I can see someone in the camera, but it doesn't quite match "
                f"the photo we have on file for {audio_name}. "
                f"Could you confirm your identity?"
            )

        return {
            "verified": verified,
            "distance": round(distance, 4),
            "message": message,
            "has_photo": True,
            "employee_id": employee.id,
        }

    except Exception as e:
        logger.error("DeepFace verification failed for '%s': %s", audio_name, e)
        # On any unexpected error, be permissive — don't block the employee
        return {
            "verified": True,
            "distance": -1.0,
            "message": "",
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
