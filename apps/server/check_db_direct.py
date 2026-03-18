import sqlite3
import os

# IMPORTANT: Double check this path matches where your office.db is!
db_path = "office.db"


def test_connection():
    print("--- Starting Database Check ---")

    if not os.path.exists(db_path):
        print(f"ERROR: Database file NOT FOUND at: {db_path}")
        return

    try:
        # Connect directly to the SQLite file
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("SUCCESS: Connected to database file.")

        # Try to find Rohit
        cursor.execute(
            "SELECT name, role, cabin_number FROM employees WHERE name LIKE '%Rohit%'"
        )
        row = cursor.fetchone()

        if row:
            print(f"DATA FOUND -> Name: {row[0]}, Role: {row[1]}, Cabin: {row[2]}")
        else:
            print(
                "RESULT: DB Connected, but 'Rohit' was not found in the 'employees' table."
            )

        conn.close()
    except Exception as e:
        print(f"DATABASE ERROR: {e}")


if __name__ == "__main__":
    test_connection()
