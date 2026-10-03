"""Lookup tools. Each store is replaceable without changing the tool names."""

from __future__ import annotations

from langchain_core.tools import tool

from triage_agent.tools.knowledge_base import JsonKnowledgeBase, KnowledgeBase
from triage_agent.tools.ticket_db import (
    CustomerStore,
    JsonCustomerStore,
    JsonOrderStore,
    JsonPastTicketStore,
    OrderStore,
    PastTicketStore,
)

__all__ = [
    "CustomerStore",
    "JsonCustomerStore",
    "JsonKnowledgeBase",
    "JsonOrderStore",
    "JsonPastTicketStore",
    "KnowledgeBase",
    "OrderStore",
    "PastTicketStore",
    "build_tools",
]


def build_tools(
    customers: CustomerStore | None = None,
    orders: OrderStore | None = None,
    past_tickets: PastTicketStore | None = None,
    knowledge_base: KnowledgeBase | None = None,
) -> list:
    """LangChain tools over the four lookup interfaces."""
    customer_store = customers if customers is not None else JsonCustomerStore()
    order_store = orders if orders is not None else JsonOrderStore()
    past_store = past_tickets if past_tickets is not None else JsonPastTicketStore()
    kb_store = knowledge_base if knowledge_base is not None else JsonKnowledgeBase()

    @tool
    def get_customer(customer_id: str) -> str:
        """Look up one customer by id. found is false when the id is unknown."""
        return customer_store.get_customer(customer_id).model_dump_json()

    @tool
    def get_orders(customer_id: str) -> str:
        """List orders for one customer. An empty list means none are on file."""
        return order_store.get_orders(customer_id).model_dump_json()

    @tool
    def search_past_tickets(customer_id: str, query: str) -> str:
        """Find this customer's past tickets by keyword. No hits is a normal result."""
        return past_store.search_past_tickets(customer_id, query).model_dump_json()

    @tool
    def search_kb(query: str) -> str:
        """Find knowledge-base articles by keyword. No hits is a normal result."""
        return kb_store.search_kb(query).model_dump_json()

    return [get_customer, get_orders, search_past_tickets, search_kb]
