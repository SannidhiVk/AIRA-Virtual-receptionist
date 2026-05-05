"""
migrate_visitors.py
-------------------
One-time migration: adds `first_seen` and `last_seen` columns to the
`visitors` table in office.db.

Run this ONCE from the project root (next to office.db), or from inside
apps/server/ depending on where your office.db lives:

    python migrate_visitors.py

It is safe to run multiple times — it checks whether each column already
exists before trying to add it.
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Locate office.db ──────────────────────────────────────────────────────────
# Adjust this path if your office.db is somewhere else.
# database.py places it next to itself (inside receptionist/ or apps/server/).
DB_PATH = Path(__file__).resolve().parent / "office.db"

if not DB_PATH.exists():
    # Try one level up (if you run the script from a subdirectory)
    DB_PATH = Path(__file__).resolve().parent.parent / "office.db"

if not DB_PATH.exists():
    raise FileNotFoundError(
        f"Could not find office.db. Looked in:\n"
        f"  {Path(__file__).resolve().parent / 'office.db'}\n"
        f"  {Path(__file__).resolve().parent.parent / 'office.db'}\n"
        f"Set DB_PATH manually at the top of this script."
    )

logger.info("Using database at: %s", DB_PATH)

# ── Migration ─────────────────────────────────────────────────────────────────
NOW = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

COLUMNS_TO_ADD = [
    # (column_name, column_definition, default_value_for_existing_rows)
    ("first_seen", "DATETIME", NOW),
    ("last_seen", "DATETIME", NOW),
]

conn = sqlite3.connect(str(DB_PATH))
try:
    cursor = conn.cursor()

    # Fetch existing columns
    cursor.execute("PRAGMA table_info(visitors)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    logger.info("Existing columns in 'visitors': %s", existing_columns)

    for col_name, col_type, default_val in COLUMNS_TO_ADD:
        if col_name in existing_columns:
            logger.info("Column '%s' already exists — skipping.", col_name)
            continue

        # SQLite ALTER TABLE only supports ADD COLUMN (no DEFAULT on NOT NULL cols).
        # We add the column as nullable, then back-fill existing rows.
        cursor.execute(f"ALTER TABLE visitors ADD COLUMN {col_name} {col_type}")
        cursor.execute(
            f"UPDATE visitors SET {col_name} = ? WHERE {col_name} IS NULL",
            (default_val,),
        )
        conn.commit()
        logger.info(
            "✅ Added column '%s' and back-filled %d row(s).", col_name, cursor.rowcount
        )

    logger.info("Migration complete.")

except Exception as e:
    conn.rollback()
    logger.error("Migration failed: %s", e)
    raise
finally:
    conn.close()
