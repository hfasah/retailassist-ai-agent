import json

from retailassist.store import ORDERS, PRODUCTS
from retailassist.tools import get_order_status, get_product_details, get_store_policy, search_products


def test_data_loaded():
    assert len(PRODUCTS) == 30
    assert len(ORDERS) == 100


def test_search_finds_matching_products():
    results = json.loads(search_products.call({"query": "hiking"}))
    skus = {r["sku"] for r in results}
    assert {"SKU-1001", "SKU-3001"} <= skus


def test_search_respects_max_price():
    results = json.loads(search_products.call({"query": "hiking", "max_price": 50}))
    assert results and all(r["price"] <= 50 for r in results)


def test_search_with_no_matches():
    assert search_products.call({"query": "kayak"}) == "No products matched that search."


def test_product_details_and_unknown_sku():
    assert json.loads(get_product_details.call({"sku": "sku-2001"}))["name"] == "Summit 30L Daypack"
    assert "No product found" in get_product_details.call({"sku": "SKU-9999"})


def test_order_lookup_requires_matching_email():
    order = ORDERS["ORD-5001"]
    email = order["customer"]["email"]
    found = json.loads(get_order_status.call({"order_id": "ord-5001", "email": email.upper()}))
    assert found["order_id"] == "ORD-5001"
    assert "No order matches" in get_order_status.call({"order_id": "ORD-5001", "email": "someone@example.com"})
    assert "No order matches" in get_order_status.call({"order_id": "ORD-0000", "email": email})


def test_order_totals_add_up():
    for order in ORDERS.values():
        subtotal = round(sum(i["quantity"] * i["unit_price"] for i in order["items"]), 2)
        assert order["subtotal"] == subtotal
        assert order["total"] == round(subtotal + order["shipping_cost"] + order["tax"], 2)


def test_store_policy():
    assert "30 days of delivery" in get_store_policy.call({"topic": "refunds"})
    assert "$75" in get_store_policy.call({"topic": "shipping"})
    schema = get_store_policy.to_dict()["input_schema"]["properties"]["topic"]
    assert schema["enum"] == ["refunds", "shipping"]


def test_create_support_ticket():
    from retailassist.tickets import ticket_store
    from retailassist.tools import create_support_ticket

    email = ORDERS["ORD-5001"]["customer"]["email"]
    result = json.loads(create_support_ticket.call(
        {"email": email, "category": "return", "summary": "Wants to return the hip pack", "order_id": "ORD-5001"}))
    ticket = ticket_store.get(int(result["ticket_id"].removeprefix("TCK-")))
    assert ticket["order_id"] == "ORD-5001" and ticket["source"] == "chat"
    assert "no ticket was created" in create_support_ticket.call(
        {"email": "someone@example.com", "category": "return", "summary": "x", "order_id": "ORD-5001"})
