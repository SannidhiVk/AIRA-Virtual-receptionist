from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    role = Column(String)
    department = Column(String)
    cabin_number = Column(String)


class Visitor(Base):
    """Logs every person who talks to the receptionist."""

    __tablename__ = "visitors"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    status = Column(String)  # e.g., "New Intern", "Candidate", "Guest"
    checkin_time = Column(DateTime, default=datetime.utcnow)


class Meeting(Base):
    """Logs specific scheduled meetings: Who, Whom, and When."""

    __tablename__ = "meetings"  # Different name to avoid the crash!
    id = Column(Integer, primary_key=True)
    visitor_name = Column(String)
    employee_name = Column(String)
    scheduled_time = Column(DateTime)
    status = Column(String, default="Scheduled")
