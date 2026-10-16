import json

from retailassist.tools import get_order_status, get_product_details, search_products


def test_search_finds_matching_products():
    results = json.loads(search_products.call({"query": "hiking"}))
    skus = {r["sku"] for r in results}
    assert {"SKU-1001", "SKU-3001"} <= skus


def test_search_respects_max_price():
    results = json.loads(search_products.call({"query": "hiking", "max_price": 50}))
    assert all(r["price"] <= 50 for r in results)


def test_search_with_no_matches():
    assert search_products.call({"query": "kayak"}) == "No products matched that search."


def test_product_details_and_unknown_sku():
    assert json.loads(get_product_details.call({"sku": "sku-2001"}))["stock"] == 7
    assert "No product found" in get_product_details.call({"sku": "SKU-9999"})


def test_order_status():
    order = json.loads(get_order_status.call({"order_id": "ORD-5001"}))
    assert order["status"] == "shipped"
    assert order["items"] == ["Trailblazer Waterproof Hiking Boots"]
    assert "No order found" in get_order_status.call({"order_id": "ORD-0000"})


def test_tool_schemas_are_generated():
    schema = search_products.to_dict()
    assert schema["name"] == "search_products"
    assert schema["input_schema"]["required"] == ["query"]
