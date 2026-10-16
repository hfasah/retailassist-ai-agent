"""Tools the agent can call. Each returns a string that is sent back to Claude."""

import json

from anthropic import beta_tool

from .store import ORDERS, PRODUCTS


@beta_tool
def search_products(query: str, max_price: float | None = None) -> str:
    """Search the product catalog by keyword.

    Args:
        query: Words to match against product names, categories, and descriptions, e.g. "hiking boots".
        max_price: Optional highest price in USD to include.
    """
    words = query.lower().split()
    matches = []
    for sku, product in PRODUCTS.items():
        text = f"{product['name']} {product['category']} {product['description']}".lower()
        if not any(word in text for word in words):
            continue
        if max_price is not None and product["price"] > max_price:
            continue
        matches.append({"sku": sku, "name": product["name"], "price": product["price"], "in_stock": product["stock"] > 0})
    if not matches:
        return "No products matched that search."
    return json.dumps(matches)


@beta_tool
def get_product_details(sku: str) -> str:
    """Get full details for one product, including price and how many are in stock.

    Args:
        sku: The product's SKU, e.g. "SKU-1001".
    """
    product = PRODUCTS.get(sku.upper())
    if product is None:
        return f"No product found with SKU {sku}."
    return json.dumps({"sku": sku.upper(), **product})


@beta_tool
def get_order_status(order_id: str) -> str:
    """Look up the status of a customer's order.

    Args:
        order_id: The order number, e.g. "ORD-5001".
    """
    order = ORDERS.get(order_id.upper())
    if order is None:
        return f"No order found with ID {order_id}. Ask the customer to double-check the number."
    items = [PRODUCTS[sku]["name"] for sku in order["items"]]
    return json.dumps({"order_id": order_id.upper(), **order, "items": items})


ALL_TOOLS = [search_products, get_product_details, get_order_status]
