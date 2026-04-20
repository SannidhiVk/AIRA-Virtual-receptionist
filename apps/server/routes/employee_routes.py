"""
employee_routes.py
------------------
REST API routes for managing employee profile photos.

ENDPOINTS:
  GET  /employees/            → List all employees (used by the admin UI)
  POST /employees/{id}/photo  → Upload a photo for an employee
  GET  /employees/{id}/photo  → Serve the stored photo as an image

HOW PHOTO STORAGE WORKS:
  - Photos are saved as JPEG files on disk:
      apps/server/receptionist/photos/employees/{employee_id}.jpg
  - The DB column `employees.photo_path` stores the relative path string.
  - The DB never stores binary image data (that would make the SQLite file huge).

FLOW (setting up an employee photo):
  1. Admin opens /admin/employees in the browser
  2. Picks an employee → clicks "Upload Photo"
  3. Browser POSTs the image to POST /employees/{id}/photo
  4. Server saves the file to disk
  5. Server updates employee.photo_path in the DB
  6. Done — that path is used by face_recognition_service.py for comparisons
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from receptionist.database import SessionLocal
from receptionist.models import Employee
from services.face_recognition_service import PHOTOS_DIR, get_photo_path, _ensure_photos_dir

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/employees", tags=["employees"])


# ── DB session dependency ─────────────────────────────────────────────────────

def get_db():
    """FastAPI dependency that provides a DB session and cleans up after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Pydantic response schema ──────────────────────────────────────────────────

class EmployeeOut(BaseModel):
    id: int
    name: str
    email: Optional[str] = None
    department: Optional[str] = None
    role: Optional[str] = None
    location: Optional[str] = None
    has_photo: bool

    class Config:
        from_attributes = True


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/", response_model=List[EmployeeOut])
def list_employees(db: Session = Depends(get_db)):
    """
    GET /employees/
    Returns all public employees with a flag indicating if a photo is stored.
    Used by the admin UI to show the employee list and photo upload status.
    """
    employees = db.query(Employee).filter(Employee.is_public == True).all()

    result = []
    for emp in employees:
        has_photo = bool(emp.photo_path and get_photo_path(emp.id).exists())
        result.append(
            EmployeeOut(
                id=emp.id,
                name=emp.name,
                email=emp.email,
                department=emp.department,
                role=emp.role,
                location=emp.location,
                has_photo=has_photo,
            )
        )
    return result


@router.post("/{employee_id}/photo")
async def upload_employee_photo(
    employee_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    POST /employees/{id}/photo
    Upload a face photo for an employee.

    HOW IT WORKS:
    1. Validates the employee exists
    2. Validates the file is an image (jpg, jpeg, png)
    3. Saves the file to disk at receptionist/photos/employees/{id}.jpg
    4. Updates employee.photo_path in the DB
    5. Returns success confirmation

    IMPORTANT: Always saves as .jpg regardless of input format.
    DeepFace works best with JPEG, and standardising the extension
    makes the lookup in face_recognition_service.py simple and reliable.
    """
    # 1. Check employee exists
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")

    # 2. Validate file type
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only JPEG/PNG/WebP allowed.",
        )

    # 3. Ensure the photos directory exists
    _ensure_photos_dir()
    save_path = get_photo_path(employee_id)  # → .../photos/employees/{id}.jpg

    # 4. Save file to disk
    try:
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error("Failed to save photo for employee %d: %s", employee_id, e)
        raise HTTPException(status_code=500, detail="Failed to save photo to disk")

    # 5. Update DB: store relative path string (portable, not tied to a machine's full path)
    relative_path = f"receptionist/photos/employees/{employee_id}.jpg"
    employee.photo_path = relative_path
    db.commit()

    logger.info(
        "Stored photo for employee %d (%s) at %s",
        employee_id, employee.name, save_path,
    )

    return {
        "success": True,
        "employee_id": employee_id,
        "employee_name": employee.name,
        "photo_path": relative_path,
        "message": f"Photo uploaded successfully for {employee.name}",
    }


@router.get("/{employee_id}/photo")
def serve_employee_photo(employee_id: int, db: Session = Depends(get_db)):
    """
    GET /employees/{id}/photo
    Serves the stored photo file as an image response.
    Used by the admin UI to display photo thumbnails.

    Returns 404 if no photo has been uploaded yet.
    """
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")

    photo_path = get_photo_path(employee_id)
    if not photo_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No photo found for employee {employee_id}. Upload one first.",
        )

    return FileResponse(
        path=str(photo_path),
        media_type="image/jpeg",
        filename=f"employee_{employee_id}.jpg",
    )
