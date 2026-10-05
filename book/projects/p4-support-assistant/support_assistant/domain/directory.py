# path: book/projects/p4-support-assistant/support_assistant/domain/directory.py
"""Synthetic employee directory and the staff accounts that use the assistant."""
from __future__ import annotations

from pydantic import BaseModel
from toolkit import ToolContext


class Employee(BaseModel):
    id: str
    name: str
    email: str
    title: str
    team: str
    tenant: str            # retail | logistics | shared
    location: str
    # Present in the system of record, never returned by the tool.
    home_address: str = ""
    salary_band: str = ""

    def public_view(self) -> dict[str, str]:
        return {"id": self.id, "name": self.name, "email": self.email, "title": self.title,
                "team": self.team, "tenant": self.tenant, "location": self.location}


EMPLOYEES: list[Employee] = [
    Employee(id="E1001", name="Priya Raman", email="priya.raman@northwind.example", title="Store Manager",
             team="Store 0412", tenant="retail", location="Leeds", home_address="(redacted)", salary_band="M2"),
    Employee(id="E1002", name="Tomas Lind", email="tomas.lind@northwind.example", title="POS Engineer",
             team="Retail Platforms", tenant="retail", location="Remote", salary_band="E3"),
    Employee(id="E1003", name="Grace Okafor", email="grace.okafor@northwind.example", title="Dispatcher",
             team="North Depot", tenant="logistics", location="York", salary_band="O2"),
    Employee(id="E1004", name="Marek Novak", email="marek.novak@northwind.example", title="Warehouse Lead",
             team="Central Depot", tenant="logistics", location="Derby", salary_band="O3"),
    Employee(id="E1005", name="Ana Silva", email="ana.silva@northwind.example", title="Service Desk Agent",
             team="IT Service Desk", tenant="shared", location="Manchester", salary_band="S2"),
    Employee(id="E1006", name="Sam Chen", email="sam.chen@northwind.example", title="Service Desk Lead",
             team="IT Service Desk", tenant="shared", location="Manchester", salary_band="S4"),
]


class StaffAccount(BaseModel):
    """An authenticated user of the assistant. In production this comes from the IdP token."""

    user_id: str
    employee_id: str
    tenant: str
    groups: list[str]
    scopes: list[str]

    def to_context(self, session_id: str | None = None, request_id: str | None = None) -> ToolContext:
        return ToolContext(user_id=self.user_id, tenant=self.tenant, groups=frozenset(self.groups),
                           scopes=frozenset(self.scopes), session_id=session_id, request_id=request_id)


AGENT_SCOPES = ["tickets:read", "tickets:write", "directory:read", "status:read", "replies:draft", "replies:send"]

STAFF: dict[str, StaffAccount] = {
    "ana": StaffAccount(user_id="ana", employee_id="E1005", tenant="retail", groups=["all", "support"],
                        scopes=AGENT_SCOPES),
    "sam": StaffAccount(user_id="sam", employee_id="E1006", tenant="retail", groups=["all", "support", "support-leads"],
                        scopes=[*AGENT_SCOPES, "replies:approve"]),
    "lee": StaffAccount(user_id="lee", employee_id="E1003", tenant="logistics", groups=["all", "support-readonly"],
                        scopes=["tickets:read", "status:read"]),
    "kai": StaffAccount(user_id="kai", employee_id="E1004", tenant="logistics", groups=["all", "support", "contractor"],
                        scopes=AGENT_SCOPES),
}


class Directory:
    def __init__(self, employees: list[Employee] | None = None) -> None:
        self._by_id = {e.id: e for e in (employees or EMPLOYEES)}

    def search(self, query: str, *, tenant: str, limit: int = 5) -> list[Employee]:
        """Tenant-scoped: callers see their tenant plus shared staff, never the other tenant."""
        q = query.strip().lower()
        hits = [e for e in self._by_id.values()
                if e.tenant in (tenant, "shared")
                and (q == e.id.lower() or q in e.name.lower() or q in e.email.lower() or q in e.team.lower())]
        return hits[:limit]


__all__ = ["Employee", "EMPLOYEES", "StaffAccount", "STAFF", "AGENT_SCOPES", "Directory"]
