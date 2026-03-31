"""
seed_data.py
------------
Populates the SQLite database with initial staff, visitor, and meeting data.
Called from the server lifespan hook at startup.
"""

from datetime import datetime

from .database import SessionLocal
from .models import Employee, Meeting, Visitor


def seed_database() -> None:
    """
    Seed the receptionist SQLite DB using SQLAlchemy models.
    No-op if employee records already exist (prevents reseeding on restart).
    """
    db = SessionLocal()
    try:
        if db.query(Employee).first():
            return  # Already seeded

        # ── Staff ──────────────────────────────────────────────────────────
        employees = [
            Employee(
                name="Arjun",
                department="HR",
                role="HR Manager",
                cabin_number="201",
                floor="2",
                extension="201",
                email="arjun@sharpsoftware.in",
                is_public=1,
            ),
            Employee(
                name="Priya",
                department="HR",
                role="HR Executive",
                cabin_number="202",
                floor="2",
                extension="202",
                email="priya@sharpsoftware.in",
                is_public=1,
            ),
            Employee(
                name="Meera",
                department="Finance",
                role="Financial Analyst",
                cabin_number="305",
                floor="3",
                extension="305",
                email="meera@sharpsoftware.in",
                is_public=1,
            ),
            Employee(
                name="Rohit",
                department="Engineering",
                role="Software Engineer",
                cabin_number="110",
                floor="1",
                extension="110",
                email="rohit@sharpsoftware.in",
                is_public=1,
            ),
            Employee(
                name="Neha",
                department="Engineering",
                role="Backend Engineer",
                cabin_number="112",
                floor="1",
                extension="112",
                email="neha@sharpsoftware.in",
                is_public=1,
            ),
            Employee(
                name="Vivek",
                department="Engineering",
                role="DevOps Engineer",
                cabin_number="115",
                floor="1",
                extension="115",
                email="vivek@sharpsoftware.in",
                is_public=1,
            ),
            Employee(
                name="Kavya",
                department="Marketing",
                role="Marketing Manager",
                cabin_number="402",
                floor="4",
                extension="402",
                email="kavya@sharpsoftware.in",
                is_public=1,
            ),
            Employee(
                name="Sanjay",
                department="Admin",
                role="Office Administrator",
                cabin_number="101",
                floor="1",
                extension="101",
                email="sanjay@sharpsoftware.in",
                is_public=1,
            ),
            Employee(
                name="Aman",
                department="Sales",
                role="Sales Manager",
                cabin_number="501",
                floor="5",
                extension="501",
                email="aman@sharpsoftware.in",
                is_public=1,
            ),
            Employee(
                name="Ritu",
                department="Support",
                role="Customer Support Executive",
                cabin_number="120",
                floor="1",
                extension="120",
                email="ritu@sharpsoftware.in",
                is_public=1,
            ),
        ]

        # ── Visitors ───────────────────────────────────────────────────────
        now_iso = datetime.utcnow().isoformat()
        visitors = [
            Visitor(
                name="Sam",
                status="New Intern",
                meeting_with="Arjun",
                purpose="Onboarding",
                check_in_time=now_iso,
                checkin_time=datetime.utcnow(),
            ),
            Visitor(
                name="John",
                status="Guest",
                meeting_with="Kavya",
                purpose="Partnership discussion",
                check_in_time=now_iso,
                checkin_time=datetime.utcnow(),
            ),
            Visitor(
                name="Alice",
                status="Candidate",
                meeting_with="Priya",
                purpose="Job interview",
                check_in_time=now_iso,
                checkin_time=datetime.utcnow(),
            ),
        ]

        # ── Sample meetings ────────────────────────────────────────────────
        today = datetime.now().date().isoformat()
        meetings = [
            Meeting(
                organizer_name="Rahul",
                organizer_type="visitor",
                visitor_name="Rahul",
                employee_name="Arjun",
                meeting_date=today,
                meeting_time="10:00",
                scheduled_time=datetime.now(),
                status="scheduled",
                created_at=now_iso,
            ),
            Meeting(
                organizer_name="Anita",
                organizer_type="visitor",
                visitor_name="Anita",
                employee_name="Meera",
                meeting_date=today,
                meeting_time="11:00",
                scheduled_time=datetime.now(),
                status="completed",
                created_at=now_iso,
            ),
            Meeting(
                organizer_name="Kiran",
                organizer_type="visitor",
                visitor_name="Kiran",
                employee_name="Rohit",
                meeting_date=today,
                meeting_time="14:00",
                scheduled_time=datetime.now(),
                status="scheduled",
                created_at=now_iso,
            ),
        ]

        db.add_all(employees)
        db.add_all(visitors)
        db.add_all(meetings)
        db.commit()
    finally:
        db.close()
