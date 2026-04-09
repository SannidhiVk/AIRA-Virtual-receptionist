"""
models.py
---------
Normalized SQLAlchemy models for the receptionist application.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Employee(Base):
    """Office staff directory (Entity)."""
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    department = Column(String)
    role = Column(String)
    # Merged 'cabin_number' and 'floor' into a single workspace/location field
    location = Column(String) 
    extension = Column(String)
    reports_to = Column(Integer, ForeignKey("employees.id"), nullable=True)
    is_public = Column(Boolean, default=True)
    photo_path = Column(String)

    # Relationships
    manager = relationship("Employee", remote_side=[id], backref="subordinates")
    meetings = relationship("Meeting", back_populates="host_employee")
    logs = relationship("ReceptionLog", back_populates="employee")


class Visitor(Base):
    """External person profile (Entity). No visit-specific data here."""
    __tablename__ = "visitors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=True)
    phone = Column(String, nullable=True)
    id_photo_path = Column(String)

    # Relationships
    meetings = relationship("Meeting", back_populates="visitor")
    logs = relationship("ReceptionLog", back_populates="visitor")


class Meeting(Base):
    """Scheduled meetings (Event). Links Employee and Visitor via IDs."""
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    host_employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    visitor_id = Column(Integer, ForeignKey("visitors.id"), nullable=True)
    
    scheduled_start = Column(DateTime, nullable=False)
    scheduled_end = Column(DateTime, nullable=True)
    purpose = Column(String)
    status = Column(String, default="scheduled") # e.g., 'scheduled', 'cancelled', 'completed'
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    host_employee = relationship("Employee", back_populates="meetings")
    visitor = relationship("Visitor", back_populates="meetings")


class ReceptionLog(Base):
    """Unified log for physical check-ins/check-outs at the front desk (Event)."""
    __tablename__ = "reception_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    visitor_id = Column(Integer, ForeignKey("visitors.id"), nullable=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    
    person_type = Column(String, nullable=False) # 'VISITOR', 'EMPLOYEE', 'DELIVERY'
    badge_id = Column(String, unique=True, nullable=True) # Badge belongs to the visit, not the person
    
    check_in_time = Column(DateTime, default=datetime.utcnow)
    check_out_time = Column(DateTime, nullable=True)
    purpose = Column(String)
    notes = Column(Text)

    # Relationships
    visitor = relationship("Visitor", back_populates="logs")
    employee = relationship("Employee", back_populates="logs")

class Settings(Base):
    """Key-value application settings (Company details, config, etc.)."""
    __tablename__ = "settings"

    key = Column(String, primary_key=True)
    value = Column(Text)