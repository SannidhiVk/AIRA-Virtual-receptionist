"""
receptionist package
--------------------
Public surface for the receptionist database layer.
Import from here rather than from the sub-modules directly.
"""

from .database import (
    # Schema init
    init_db,
    # Reception log
    log_reception_entry,
    log_reception_checkout,
    # Settings
    set_setting,
    get_setting,
    # Visitors
    add_visitor,
    checkout_visitor,
    get_visitor_by_name,
    # Employees
    get_employee_by_name,
    get_employee_by_name_or_role,
    get_employee_by_name_and_department,
    get_similar_employee,
    get_hr,
    get_department_manager,
    # Meetings
    get_employee_meetings,
    get_available_slots,
    schedule_meeting,
    cancel_meeting,
    # Low-level
    SessionLocal,
    engine,
)
from .models import (
    Base,
    Employee,
    Meeting,
    ReceptionLog,
    Settings,
    Visitor,
)


__all__ = [
    # database helpers
    "init_db",
    "log_reception_entry",
    "log_reception_checkout",
    "set_setting",
    "get_setting",
    "add_visitor",
    "checkout_visitor",
    "get_visitor_by_name",
    "get_employee_by_name",
    "get_employee_by_name_or_role",
    "get_employee_by_name_and_department",
    "get_similar_employee",
    "get_hr",
    "get_department_manager",
    "get_employee_meetings",
    "get_available_slots",
    "schedule_meeting",
    "cancel_meeting",
    "SessionLocal",
    "engine",
    # models
    "Base",
    "Employee",
    "Meeting",
    "ReceptionLog",
    "Settings",
    "Visitor",
    # seeder
    "seed_database",
]
