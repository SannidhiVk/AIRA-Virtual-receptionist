"""
seed_data.py
------------
Seeds the normalized database with initial mock data, including 12 employees.
"""

from datetime import datetime
from receptionist.database import SessionLocal, init_db
from receptionist.models import Employee, Settings

def seed_database():
    init_db()
    session = SessionLocal()

    try:
        # Check if already seeded
        if session.query(Employee).first():
            print("Database already seeded.")
            return

        print("Seeding initial company settings...")
        settings = [
            Settings(key="company_name", value="Sharp Software Development India Private Limited"),
            Settings(key="company_address", value="123 Innovation Drive, Tech Park"),
            Settings(key="company_phone", value="+91-80-5555-0199"),
            Settings(key="company_email", value="contact@sharpsoftware.in"),
            Settings(key="company_website", value="www.sharpsoftware.in"),
        ]
        session.add_all(settings)

        print("Seeding 12 employees...")
        employees = [
            Employee(
                name="Priya",
                email="krutikakanchani847+priya@gmail.com",
                department="HR",
                role="HR Manager",
                location="Floor 2, Room 201",
                extension="101",
                is_public=True
            ),
            Employee(
                name="Arjun",
                email="sannidhivk2004+arjun@gmail.com",
                department="Engineering",
                role="Lead Engineer",
                location="Floor 3, Desk 35",
                extension="102",
                is_public=True
            ),
            Employee(
                name="Suresh",
                email="krutikakanchani847+suresh@gmail.com",
                department="Management",
                role="CEO",
                location="Floor 5, Executive Suite",
                extension="100",
                is_public=True
            ),
            Employee(
                name="Jack",
                email="sannidhivk2004+sannidhi@gmail.com",
                department="Sales",
                role="Sales Director",
                location="Floor 1, Room 105",
                extension="104",
                is_public=True
            ),
            Employee(
                name="john",
                email="krutikakanchani847+lavanya@gmail.com",
                department="IT Support",
                role="IT Administrator",
                location="Floor 1, Tech Bar",
                extension="110",
                is_public=True
            ),
            Employee(
                name="Virat",
                email="krutikaak07+sunil@gmail.com",
                department="Design",
                role="UX/UI Designer",
                location="Floor 2, Creative Studio",
                extension="115",
                is_public=True
            ),
            Employee(
                name="Ravi",
                email="sannidhivk2004+prajwal@gmail.com",
                department="Finance",
                role="Chief Financial Officer",
                location="Floor 5, Room 502",
                extension="120",
                is_public=True
            ),
            Employee(
                name="Rahul",
                email="sannidhivk2004+rahul@gmail.com",
                department="Marketing",
                role="Marketing Coordinator",
                location="Floor 2, Desk 12",
                extension="125",
                is_public=True
            ),
            Employee(
                name="Ramesh",
                email="sannidhivk2004+pratham@gmail.com",
                department="Engineering",
                role="Data Scientist",
                location="Floor 3, Desk 42",
                extension="130",
                is_public=True
            ),
            Employee(
                name="lucy",
                email="krutikakanchani847+shivraj@gmail.com",
                department="Legal",
                role="Legal Counsel",
                location="Floor 4, Room 410",
                extension="140",
                is_public=True
            ),
            Employee(
                name="Cookie",
                email="krutikakanchani847+vivek@gmail.com",
                department="Finance",
                role="Accountant",
                location="Floor 4, Desk 8",
                extension="145",
                is_public=True
            ),
            Employee(
                name="Jim",
                email="krutikaak07+veeresh@gmail.com",
                department="Operations",
                role="Operations Manager",
                location="Floor 1, Room 112",
                extension="150",
                is_public=True
            )
        ]
        session.add_all(employees)
        session.commit()
        
        print("Database seeded successfully with 12 employees!")

    except Exception as e:
        session.rollback()
        print(f"Error seeding database: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    seed_database()