from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Employee(Base):
    """Office staff directory."""

    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    department = Column(String)
    role = Column(String)
    email = Column(String)
    # cabin_number kept for backward-compat; maps to floor in new schema
    cabin_number = Column(String)
    floor = Column(String)
    extension = Column(String)
    reports_to = Column(Integer, ForeignKey("employees.id"), nullable=True)
    is_public = Column(Integer, default=1)  # 1 = visible in directory
    photo_path = Column(String)

    # self-referential relationship — remote_side=[id] tells SQLAlchemy that
    # 'id' is the "one" (parent) side, so 'reports_to' is the FK (many) side.
    subordinates = relationship(
        "Employee",
        backref="manager",
        foreign_keys=[reports_to],
        remote_side=[id],  # ← fix: marks 'id' as the parent/remote side
    )


class Visitor(Base):
    """Every external person who interacts with reception."""

    __tablename__ = "visitors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    # legacy status field (e.g. "New Intern", "Candidate", "Guest")
    status = Column(String)
    # richer fields from database1
    meeting_with = Column(String)
    purpose = Column(String)
    badge_id = Column(String)
    check_in_time = Column(String)  # ISO string — keeps parity with db1
    check_out_time = Column(String)
    id_photo_path = Column(String)
    # legacy DateTime column used by seed_data
    checkin_time = Column(DateTime, default=datetime.utcnow)


class Meeting(Base):
    """Scheduled meetings between visitors/employees and staff."""

    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # legacy fields (used by old database.py)
    visitor_name = Column(String)
    scheduled_time = Column(DateTime)
    # richer fields from database1
    organizer_name = Column(String)
    organizer_type = Column(String)
    employee_name = Column(String, nullable=False)
    employee_email = Column(String)
    meeting_date = Column(String)  # YYYY-MM-DD
    meeting_time = Column(String)  # HH:MM
    purpose = Column(String)
    status = Column(String, default="scheduled")
    created_at = Column(String)


class ReceptionLog(Base):
    """Unified log of every person who passed through reception."""

    __tablename__ = "reception_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_name = Column(String, nullable=False)
    # VISITOR | EMPLOYEE | DELIVERY | JOB_SEEKER
    person_type = Column(String, nullable=False)
    linked_visitor_id = Column(Integer, ForeignKey("visitors.id"), nullable=True)
    linked_employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    check_in_time = Column(String, nullable=False)
    check_out_time = Column(String)
    notes = Column(Text)


class Conversation(Base):
    """AI conversation history."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_text = Column(Text)
    ai_response = Column(Text)
    timestamp = Column(String)


class Settings(Base):
    """Key-value application settings."""

    __tablename__ = "settings"

    key = Column(String, primary_key=True)
    value = Column(Text)
