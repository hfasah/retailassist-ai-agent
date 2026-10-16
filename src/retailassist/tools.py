"""Tools the agent can call. Each returns a string that is sent back to Claude."""

import json
from typing import Literal

from anthropic import beta_tool

from .store import ORDERS, POLICIES, PRODUCTS


@beta_tool
def search_products(query: str, max_price: float | None = None) -> str:
    """Search the product catalog by keyword.

    Args:
        query: Words to match against product names, categories, and descriptions, e.g. "hiking boots".
        max_price: Optional highest price in USD to include.
    """
    words = query.lower().split()
    matches = []
    for product in PRODUCTS.values():
        text = f"{product['name']} {product['category']} {product['description']}".lower()
        if not any(word in text for word in words):
            continue
        if max_price is not None and product["price"] > max_price:
            continue
        matches.append({"sku": product["sku"], "name": product["name"], "price": product["price"],
                        "in_stock": product["stock"] > 0})
    if not matches:
        return "No products matched that search."
    return json.dumps(matches)


@beta_tool
def get_product_details(sku: str) -> str:
    """Get full details for one product, including price, stock, and whether it is final sale.

    Args:
        sku: The product's SKU, e.g. "SKU-1001".
    """
    product = PRODUCTS.get(sku.upper())
    if product is None:
        return f"No product found with SKU {sku}."
    return json.dumps(product, ensure_ascii=False)


@beta_tool
def get_order_status(order_id: str, email: str) -> str:
    """Look up a customer's order. Ask the customer for both their order number and the email used to place it.

    Args:
        order_id: The order number, e.g. "ORD-5001".
        email: The email address the order was placed with.
    """
    order = ORDERS.get(order_id.strip().upper())
    # Same message for a wrong ID or a wrong email, so the tool doesn't reveal which orders exist.
    if order is None or order["customer"]["email"] != email.strip().lower():
        return "No order matches that order number and email. Ask the customer to double-check both."
    return json.dumps(order)


@beta_tool
def get_store_policy(topic: Literal["refunds", "shipping"]) -> str:
    """Get the store's official policy text. Use this to answer questions about returns, refunds,
    exchanges, cancellations, shipping options and costs, delivery times, and tracking.

    Args:
        topic: "refunds" for the returns and refund policy, or "shipping" for the shipping FAQ.
    """
    return POLICIES[topic]


ALL_TOOLS = [search_products, get_product_details, get_order_status, get_store_policy]
