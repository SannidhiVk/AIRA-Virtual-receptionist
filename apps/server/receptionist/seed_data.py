from datetime import datetime
from .database import SessionLocal
from .models import Employee, Visitor, Meeting  # Import all three models


def seed_database():
    db = SessionLocal()

    # Prevent reseeding if data already exists
    if db.query(Employee).first():
        db.close()
        return

    # 1. Staff List
    employees = [
        Employee(name="Arjun", department="HR", cabin_number="201", role="HR Manager"),
        Employee(
            name="Meera",
            department="Finance",
            cabin_number="305",
            role="Financial Analyst",
        ),
        Employee(
            name="Rohit",
            department="Engineering",
            cabin_number="110",
            role="Software Engineer",
        ),
        Employee(
            name="Kavya",
            department="Marketing",
            cabin_number="402",
            role="Marketing Manager",
        ),
        Employee(
            name="Sanjay",
            department="Admin",
            cabin_number="101",
            role="Office Administrator",
        ),
        Employee(
            name="Neha",
            department="Engineering",
            cabin_number="112",
            role="Backend Engineer",
        ),
        Employee(
            name="Vivek",
            department="Engineering",
            cabin_number="115",
            role="DevOps Engineer",
        ),
        Employee(
            name="Priya", department="HR", cabin_number="202", role="HR Executive"
        ),
        Employee(
            name="Aman", department="Sales", cabin_number="501", role="Sales Manager"
        ),
        Employee(
            name="Ritu",
            department="Support",
            cabin_number="120",
            role="Customer Support Executive",
        ),
    ]

    # 2. General Visitor History (People who just walked in)
    visitors = [
        Visitor(name="Sam", status="New Intern", checkin_time=datetime.now()),
        Visitor(name="John", status="Guest", checkin_time=datetime.now()),
        Visitor(name="Alice", status="Candidate", checkin_time=datetime.now()),
    ]

    # 3. Scheduled Meetings (Who is meeting Whom and When)
    meetings = [
        Meeting(
            visitor_name="Rahul",
            employee_name="Arjun",
            scheduled_time=datetime.now(),
            status="Scheduled",
        ),
        Meeting(
            visitor_name="Anita",
            employee_name="Meera",
            scheduled_time=datetime.now(),
            status="Completed",
        ),
        Meeting(
            visitor_name="Kiran",
            employee_name="Rohit",
            scheduled_time=datetime.now(),
            status="Scheduled",
        ),
    ]

    # Add everything to the database
    db.add_all(employees)
    db.add_all(visitors)
    db.add_all(meetings)

    db.commit()
    db.close()
    print("Database seeded successfully with Employees, Visitors, and Meetings.")
