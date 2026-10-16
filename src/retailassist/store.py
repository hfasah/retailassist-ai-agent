"""Loads the fictional products and orders in data/. Policies are indexed by rag.py. Replace with calls to your real catalog and order systems."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

PRODUCTS = {p["sku"]: p for p in json.loads((DATA_DIR / "products.json").read_text())}
ORDERS = {o["order_id"]: o for o in json.loads((DATA_DIR / "orders.json").read_text())}


def find_order(order_id: str, email: str) -> dict | None:
    """Return the order only if the email matches the one it was placed with.

    A wrong ID and a wrong email look the same to the caller, so this doesn't reveal which orders exist.
    """
    order = ORDERS.get(order_id.strip().upper())
    if order is None or order["customer"]["email"] != email.strip().lower():
        return None
    return order
