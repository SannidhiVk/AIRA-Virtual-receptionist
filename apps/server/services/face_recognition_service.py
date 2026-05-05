"""
face_recognition_service.py
---------------------------
Shared face verification helpers for:
- employees matched against a stored DB photo
- visitors matched only against a reference frame captured in the current session

Diagnostic captures are saved only for mismatches/errors and are cleaned up based
on the configured retention window.
"""

import base64
import logging
import os
import tempfile
from datetime import datetime, timedelta
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
CAPTURE_RETENTION_DAYS = int(os.getenv("CAPTURE_RETENTION_DAYS", "7"))

# Photo storage root — relative to this file's location (apps/server/)
PHOTOS_DIR = (
    Path(__file__).resolve().parent.parent / "receptionist" / "photos" / "employees"
)

# Visitor photos — separate directory for clean separation and easier bulk cleanup
VISITOR_PHOTOS_DIR = (
    Path(__file__).resolve().parent.parent / "receptionist" / "photos" / "visitors"
)

# Captures directory — diagnostic frames saved here for mismatches/errors
CAPTURES_DIR = (
    Path(__file__).resolve().parent.parent / "receptionist" / "photos" / "captures"
)


def _ensure_photos_dir() -> None:
    """Create all photo directories if they don't exist yet."""
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    VISITOR_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
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
    image_b64: str, subject_name: str, verified: bool, distance: float, reason: str
) -> Optional[Path]:
    """
    Save a diagnostic capture to CAPTURES_DIR for mismatches or errors only.
    Routes to 'employees' or 'visitor_sessions' subfolders based on the reason.
    """
    if CAPTURE_RETENTION_DAYS <= 0:
        logger.info(
            "Capture saving disabled via CAPTURE_RETENTION_DAYS=%s",
            CAPTURE_RETENTION_DAYS,
        )
        return None

    try:
        raw_b64 = image_b64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]

        image_bytes = base64.b64decode(raw_b64)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = subject_name.replace(" ", "_")[:30] or "unknown"
        safe_reason = reason.replace(" ", "_")[:20]
        result_tag = "MATCH" if verified else "MISMATCH"
        dist_tag = f"d{distance:.3f}".replace(".", "p")
        filename = f"{ts}_{safe_name}_{safe_reason}_{result_tag}_{dist_tag}.jpg"

        # --- NEW ROUTING LOGIC ---
        if "employee" in reason.lower():
            target_dir = CAPTURES_DIR / "employees"
        elif "visitor" in reason.lower():
            target_dir = CAPTURES_DIR / "visitor_sessions"
        else:
            target_dir = CAPTURES_DIR

        # Ensure the subfolder exists before saving
        target_dir.mkdir(parents=True, exist_ok=True)

        save_path = target_dir / filename
        # -------------------------

        save_path.write_bytes(image_bytes)
        logger.info("Capture saved to: %s", save_path)
        return save_path
    except Exception as e:
        logger.warning("Could not save capture for '%s': %s", subject_name, e)
        return None


def cleanup_old_captures(max_age_days: Optional[int] = None) -> int:
    """
    Remove capture JPEGs and visitor photos older than the retention period.
    - Diagnostic captures in CAPTURES_DIR (and subdirs) → deleted after retention_days
    - Visitor photos in VISITOR_PHOTOS_DIR → deleted after retention_days + DB cleared
    Returns the number of deleted files.
    """
    _ensure_photos_dir()
    retention_days = CAPTURE_RETENTION_DAYS if max_age_days is None else max_age_days
    if retention_days <= 0:
        logger.info("Capture cleanup skipped because retention is disabled.")
        return 0

    cutoff = datetime.now() - timedelta(days=retention_days)
    deleted = 0

    # 1. Clean diagnostic captures (recurse into subdirs)
    for file_path in CAPTURES_DIR.rglob("*.jpg"):
        try:
            modified_at = datetime.fromtimestamp(file_path.stat().st_mtime)
            if modified_at < cutoff:
                file_path.unlink()
                deleted += 1
        except Exception as e:
            logger.warning("Failed to clean up capture '%s': %s", file_path, e)

    # 2. Clean old visitor photos and clear DB references
    for file_path in VISITOR_PHOTOS_DIR.glob("*.jpg"):
        try:
            modified_at = datetime.fromtimestamp(file_path.stat().st_mtime)
            if modified_at < cutoff:
                # Extract visitor ID from filename (e.g. "42.jpg" → 42)
                try:
                    visitor_id = int(file_path.stem)
                    _clear_visitor_photo_in_db(visitor_id)
                except ValueError:
                    pass
                file_path.unlink()
                deleted += 1
        except Exception as e:
            logger.warning("Failed to clean up visitor photo '%s': %s", file_path, e)

    if deleted:
        logger.info(
            "Removed %s file(s) older than %s days (captures + visitor photos).",
            deleted,
            retention_days,
        )
    return deleted


def _clear_visitor_photo_in_db(visitor_id: int) -> None:
    """Clear the id_photo_path field in the visitors table after photo cleanup."""
    try:
        from receptionist.database import SessionLocal
        from receptionist.models import Visitor

        session = SessionLocal()
        try:
            visitor = session.query(Visitor).filter(Visitor.id == visitor_id).first()
            if visitor and visitor.id_photo_path:
                visitor.id_photo_path = None
                session.commit()
                logger.info(
                    "Cleared photo path for visitor ID %d after cleanup.", visitor_id
                )
        finally:
            session.close()
    except Exception as e:
        logger.warning("Failed to clear visitor %d photo path in DB: %s", visitor_id, e)


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


def _run_face_comparison(
    reference_path: str, live_path: str, subject_name: str, mismatch_message: str
) -> dict:
    from deepface import DeepFace  # Lazy import — only loaded when first used

    logger.info(
        "Running DeepFace.verify | model=%s | detector=%s | reference=%s | live=%s",
        MODEL_NAME,
        DETECTOR_BACKEND,
        reference_path,
        live_path,
    )

    result = DeepFace.verify(
        img1_path=reference_path,
        img2_path=live_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
        distance_metric="cosine",
        align=True,
    )

    verified: bool = result.get("verified", False)
    distance: float = result.get("distance", 1.0)
    deepface_threshold = result.get("threshold", VERIFY_THRESHOLD)
    logger.info(
        "DeepFace raw result | distance=%.4f | deepface_threshold=%.4f | our_threshold=%.2f",
        distance,
        deepface_threshold,
        VERIFY_THRESHOLD,
    )

    if distance > 0.9:
        logger.warning(
            "Distance %.4f is suspiciously high (>0.9) for '%s' — face likely not detected.",
            distance,
            subject_name,
        )

    if distance > VERIFY_THRESHOLD:
        verified = False

    logger.info(
        "Face verify for '%s': verified=%s, distance=%.4f (threshold=%.2f)",
        subject_name,
        verified,
        distance,
        VERIFY_THRESHOLD,
    )

    message = (
        f"Identity verified for {subject_name}. You can proceed with your question or request."
        if verified
        else mismatch_message
    )

    return {
        "verified": verified,
        "distance": round(distance, 4),
        "message": message,
        "has_photo": True,
        "face_detected": True,
        "capture_dir": str(CAPTURES_DIR),
    }


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
            "verified": False,
            "distance": -1.0,
            "message": f"I could not find an employee record for {audio_name}. Please confirm the name and try again.",
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
            "verified": False,
            "distance": -1.0,
            "message": f"I do not have a face photo on file for {audio_name}, so I cannot verify this identity yet.",
            "has_photo": False,
            "employee_id": employee.id,
        }

    tmp_path = decode_b64_to_tempfile(image_b64)
    if not tmp_path:
        return {
            "verified": False,
            "distance": -1.0,
            "message": "I could not read the camera frame for face verification. Please try again.",
            "has_photo": True,
            "employee_id": employee.id,
        }

    try:
        result = _run_face_comparison(
            str(stored_photo_path),
            tmp_path,
            audio_name,
            (
                f"I can see someone in the camera, but it doesn't quite match "
                f"the photo we have on file for {audio_name}. "
                f"Could you confirm your identity?"
            ),
        )
        if not result["verified"]:
            _save_capture(
                image_b64, audio_name, False, result["distance"], "employee_mismatch"
            )

        return {**result, "employee_id": employee.id}

    except Exception as e:
        error_msg = str(e)
        is_no_face = (
            "Face could not be detected" in error_msg
            or "FaceNotDetected" in error_msg
            or "Exception while processing img" in error_msg
        )

        if is_no_face:
            logger.debug(
                "No face detected in frame for employee '%s' — silent, not a strike.",
                audio_name,
            )
            message = "I cannot see your face clearly. Please step into the camera frame so I can verify you."
        else:
            logger.error(
                "DeepFace verification failed for '%s': %s",
                audio_name,
                e,
                exc_info=True,
            )
            _save_capture(image_b64, audio_name, False, -1.0, "employee_error")
            message = "I'm sorry, I encountered an error while trying to verify your face. Please try again."

        return {
            "verified": False,
            "distance": -1.0,
            "face_detected": False,
            "message": message,
            "has_photo": True,
            "employee_id": employee.id,
        }

    finally:
        # Always clean up the temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def get_visitor_photo_path(visitor_id: int) -> Path:
    """Return the expected disk path for a visitor's stored photo."""
    return VISITOR_PHOTOS_DIR / f"{visitor_id}.jpg"


def save_visitor_photo_from_b64(visitor_id: int, image_b64: str) -> Optional[Path]:
    """
    Save a visitor's first-visit photo from a base64 camera frame.
    Returns the saved path, or None on failure.
    """
    _ensure_photos_dir()
    try:
        raw_b64 = image_b64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(raw_b64)
        save_path = get_visitor_photo_path(visitor_id)
        save_path.write_bytes(image_bytes)
        logger.info(
            "Saved first-visit photo for visitor ID %d at %s", visitor_id, save_path
        )
        return save_path
    except Exception as e:
        logger.error("Failed to save visitor %d photo: %s", visitor_id, e)
        return None


def verify_visitor_face(visitor_name: str, image_b64: str) -> dict:
    """
    Visitor face recognition with deduplication.

    FLOW:
    1. Look up visitor by name in DB (get_or_create_visitor)
    2. If visitor has a stored photo (returning visitor):
       → DeepFace verify live frame vs stored photo
       → If MATCH: log visit, greet "Welcome back!"
       → If MISMATCH: could be same name different person → create new visitor
    3. If visitor has NO photo (first visit):
       → Save the captured frame as their photo
       → Greet "Welcome! I've noted your visit."

    Returns:
        {
            "verified":    bool
            "distance":    float
            "is_new":      bool   — True if this is a first-time visitor
            "visitor_id":  int
            "message":     str    — Greeting text for Jarvis TTS
            "has_photo":   bool
        }
    """
    _ensure_photos_dir()

    # ── Step 1: Get or create the visitor in DB ───────────────────────────────
    visitor, is_new = _get_or_create_visitor_with_status(visitor_name)

    if not visitor:
        logger.error("Failed to get/create visitor for name '%s'", visitor_name)
        return {
            "verified": False,
            "distance": -1.0,
            "is_new": True,
            "visitor_id": None,
            "message": f"Welcome! I've noted your visit, {visitor_name}.",
            "has_photo": False,
        }

    # ── Step 2: Check if visitor has a stored photo ───────────────────────────
    stored_photo = get_visitor_photo_path(visitor.id)
    has_stored_photo = bool(visitor.id_photo_path and stored_photo.exists())

    if not has_stored_photo:
        # FIRST VISIT — save the frame as their reference photo
        saved = save_visitor_photo_from_b64(visitor.id, image_b64)
        if saved:
            _update_visitor_photo_path(
                visitor.id, f"receptionist/photos/visitors/{visitor.id}.jpg"
            )

        return {
            "verified": True,
            "distance": 0.0,
            "is_new": True,
            "visitor_id": visitor.id,
            "message": f"Welcome! I've noted your visit, {visitor_name}.",
            "has_photo": bool(saved),
        }

    # ── Step 3: RETURNING VISITOR — compare live frame against stored photo ───
    tmp_path = decode_b64_to_tempfile(image_b64)
    if not tmp_path:
        return {
            "verified": False,
            "distance": -1.0,
            "is_new": False,
            "visitor_id": visitor.id,
            "message": "I could not read the camera frame. Please try again.",
            "has_photo": True,
        }

    try:
        result = _run_face_comparison(
            str(stored_photo),
            tmp_path,
            visitor_name,
            (
                f"I see someone different from the {visitor_name} I met before. "
                f"Could you confirm your identity?"
            ),
        )

        if result["verified"]:
            # Same person returning — update last_seen, no new photo
            _bump_visitor_last_seen(visitor.id)
            result["message"] = f"Welcome back, {visitor_name}! Good to see you again."
            result["is_new"] = False
        # AFTER:
        else:
            # Same name but DIFFERENT face — create a brand new visitor record
            logger.info(
                f"Face mismatch for '{visitor_name}' (distance={result['distance']:.3f}) "
                f"— creating new visitor record."
            )
            new_visitor = _create_new_visitor(visitor_name)
            if new_visitor:
                saved = save_visitor_photo_from_b64(new_visitor.id, image_b64)
                if saved:
                    _update_visitor_photo_path(
                        new_visitor.id,
                        f"receptionist/photos/visitors/{new_visitor.id}.jpg",
                    )
                result["visitor_id"] = new_visitor.id
                result["is_new"] = True
                result["verified"] = True  # treat as fresh first-visit
                result["distance"] = 0.0
            else:
                result["visitor_id"] = visitor.id
                result["is_new"] = False

        return result

    except Exception as e:
        error_msg = str(e)
        is_no_face = (
            "Face could not be detected" in error_msg or "FaceNotDetected" in error_msg
        )
        if is_no_face:
            logger.debug("No face detected for visitor '%s'.", visitor_name)
            message = (
                "I cannot see your face clearly. Please step into the camera frame."
            )
        else:
            logger.error("DeepFace failed for visitor '%s': %s", visitor_name, e)
            _save_capture(image_b64, visitor_name, False, -1.0, "visitor_error")
            message = "I encountered an error during verification. Please try again."

        return {
            "verified": False,
            "distance": -1.0,
            "face_detected": False,
            "is_new": False,
            "visitor_id": visitor.id,
            "message": message,
            "has_photo": True,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def verify_person_face(
    *,
    person_type: str,
    image_b64: str,
    audio_name: str = "",
) -> dict:
    """
    Route verification to the correct strategy based on person_type.
    """
    if person_type == "visitor":
        return verify_visitor_face(audio_name, image_b64)
    return verify_employee_face(audio_name, image_b64)


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


def _get_or_create_visitor_with_status(name: str):
    """
    Look up a visitor by name, or create a new one.
    Returns (visitor_obj, is_new: bool).
    Uses the existing get_or_create_visitor from database.py.
    """
    try:
        from receptionist.database import SessionLocal
        from receptionist.models import Visitor

        session = SessionLocal()
        try:
            # Check if visitor already exists (case-insensitive)
            visitor = session.query(Visitor).filter(Visitor.name.ilike(name)).first()
            if visitor:
                return visitor, False
            else:
                # Create new visitor
                visitor = Visitor(name=name)
                session.add(visitor)
                session.commit()
                session.refresh(visitor)
                return visitor, True
        finally:
            session.close()
    except Exception as e:
        logger.error("Failed to get/create visitor '%s': %s", name, e)
        return None, True


def _create_new_visitor(name: str):
    """
    Always creates a NEW visitor record regardless of existing records with same name.
    Used when a face mismatch detects a different person sharing a name.
    """
    try:
        from receptionist.database import SessionLocal
        from receptionist.models import Visitor

        session = SessionLocal()
        try:
            visitor = Visitor(name=name)
            session.add(visitor)
            session.commit()
            session.refresh(visitor)
            logger.info(
                f"Created new visitor record for '{name}' (ID={visitor.id}) "
                f"due to face mismatch with existing record."
            )
            return visitor
        finally:
            session.close()
    except Exception as e:
        logger.error("Failed to create new visitor for '%s': %s", name, e)
        return None


def _update_visitor_photo_path(visitor_id: int, photo_path: str) -> None:
    """Set the id_photo_path field on a visitor after saving their first photo."""
    try:
        from receptionist.database import SessionLocal
        from receptionist.models import Visitor

        session = SessionLocal()
        try:
            visitor = session.query(Visitor).filter(Visitor.id == visitor_id).first()
            if visitor:
                visitor.id_photo_path = photo_path
                session.commit()
                logger.info("Set photo path for visitor %d: %s", visitor_id, photo_path)
        finally:
            session.close()
    except Exception as e:
        logger.error("Failed to update photo path for visitor %d: %s", visitor_id, e)


def _bump_visitor_last_seen(visitor_id: int) -> None:
    """Update the last_seen timestamp for a returning visitor."""
    try:
        from receptionist.database import SessionLocal
        from receptionist.models import Visitor

        session = SessionLocal()
        try:
            visitor = session.query(Visitor).filter(Visitor.id == visitor_id).first()
            if visitor:
                visitor.last_seen = datetime.utcnow()
                session.commit()
                logger.info("Bumped last_seen for visitor %d.", visitor_id)
        finally:
            session.close()
    except Exception as e:
        logger.error("Failed to bump last_seen for visitor %d: %s", visitor_id, e)
