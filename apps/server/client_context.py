"""
client_context.py
-----------------
Shared per-client context store.

Allows different parts of the server (main.py employee lookups,
meeting_scheduler.py slot-filling) to share state within a single
client session — specifically so that a resolved employee from a
lookup ("Who is the sales manager?") is remembered when the next
utterance uses a pronoun ("Schedule a meeting with him/her").
"""

from typing import Any, Dict, Optional

# { client_id: { "last_employee_name": str, "last_employee_email": str, ... } }
_client_context: Dict[str, Dict[str, Any]] = {}


def get_context(client_id: str) -> Dict[str, Any]:
    if client_id not in _client_context:
        _client_context[client_id] = {}
    return _client_context[client_id]


def set_last_employee(
    client_id: str,
    name: str,
    email: Optional[str] = None,
    cabin: Optional[str] = None,
    role: Optional[str] = None,
    department: Optional[str] = None,
) -> None:
    """Called after any successful employee lookup so the result can be
    referenced by pronouns in the next turn."""
    ctx = get_context(client_id)
    ctx["last_employee_name"] = name
    ctx["last_employee_email"] = email
    ctx["last_employee_cabin"] = cabin
    ctx["last_employee_role"] = role
    ctx["last_employee_department"] = department


def get_last_employee_name(client_id: str) -> Optional[str]:
    return get_context(client_id).get("last_employee_name")


def clear_context(client_id: str) -> None:
    _client_context.pop(client_id, None)