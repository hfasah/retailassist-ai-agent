"""Loads the fictional store data in data/. Replace with calls to your real catalog and order systems."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

PRODUCTS = {p["sku"]: p for p in json.loads((DATA_DIR / "products.json").read_text())}
ORDERS = {o["order_id"]: o for o in json.loads((DATA_DIR / "orders.json").read_text())}
POLICIES = {
    "refunds": (DATA_DIR / "policies" / "refund-policy.md").read_text(),
    "shipping": (DATA_DIR / "policies" / "shipping-faq.md").read_text(),
}
