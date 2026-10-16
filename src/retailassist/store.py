"""Loads the fictional products and orders in data/. Policies are indexed by rag.py. Replace with calls to your real catalog and order systems."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

PRODUCTS = {p["sku"]: p for p in json.loads((DATA_DIR / "products.json").read_text())}
ORDERS = {o["order_id"]: o for o in json.loads((DATA_DIR / "orders.json").read_text())}
