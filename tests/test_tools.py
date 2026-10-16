import pytest

from retailassist.store import ORDERS, PRODUCTS
from retailassist.tools import (ToolInputError, get_order_status, get_product_details, search_products,
                                search_store_policies, validate_args)


def test_data_loaded():
    assert len(PRODUCTS) == 30
    assert len(ORDERS) == 100


def test_search_finds_matching_products():
    skus = {r.sku for r in search_products.call({"query": "hiking"}).products}
    assert {"SKU-1001", "SKU-3001"} <= skus


def test_search_respects_max_price():
    results = search_products.call({"query": "hiking", "max_price": 50}).products
    assert results and all(r.price <= 50 for r in results)


def test_search_with_no_matches():
    result = search_products.call({"query": "kayak"})
    assert result.products == [] and result.message == "No products matched that search."


def test_product_details_and_unknown_sku():
    assert get_product_details.call({"sku": "sku-2001"}).name == "Summit 30L Daypack"
    with pytest.raises(ValueError, match="No product found"):
        get_product_details.call({"sku": "SKU-9999"})


def test_order_lookup_requires_matching_email():
    order = ORDERS["ORD-5001"]
    email = order["customer"]["email"]
    found = get_order_status.call({"order_id": "ord-5001", "email": email.upper()})
    assert found.order_id == "ORD-5001" and found.customer_name == order["customer"]["name"]
    assert "customer" not in found.model_dump()  # the stored email and contact details stay out
    with pytest.raises(ValueError, match="No order matches"):
        get_order_status.call({"order_id": "ORD-5001", "email": "someone@example.com"})


# --- Input validation -----------------------------------------------------------------------------

@pytest.mark.parametrize("name,args,expected", [
    ("get_order_status", {"order_id": " ord-5003 ", "email": " Aisha@Example.com"},
     {"order_id": "ORD-5003", "email": "aisha@example.com"}),
    ("search_products", {"query": "  tent ", "max_price": "50"}, {"query": "tent", "max_price": 50.0}),
    ("request_refund", {"order_id": "ORD-5003", "email": "a@example.com", "reason": "defective", "skus": ["sku-1001"]},
     {"order_id": "ORD-5003", "email": "a@example.com", "reason": "defective", "skus": ["SKU-1001"], "details": ""}),
])
def test_validate_args_normalizes(name, args, expected):
    assert validate_args(name, args) == expected


@pytest.mark.parametrize("name,args,message", [
    ("get_order_status", {"order_id": "5003", "email": "a@example.com"}, "order_id: should look like ORD-5001"),
    ("get_order_status", {"order_id": 5003, "email": "a@example.com"}, "order_id: Input should be a valid string"),
    ("get_order_status", {"order_id": "ORD-5003"}, "email: required"),
    ("get_order_status", {"order_id": "ORD-5003", "email": "nope"}, "email: should look like name@example.com"),
    ("search_products", {"query": ""}, "query: String should have at least 1 character"),
    ("search_products", {"query": "tent", "max_price": -5}, "max_price: Input should be greater than 0"),
    ("search_products", {"query": "tent", "colour": "red"}, "colour: Extra inputs are not permitted"),
    ("get_product_details", {"sku": "boots"}, "sku: should look like SKU-1001"),
    ("request_refund", {"order_id": "ORD-5003", "email": "a@example.com", "reason": "free money"},
     "reason: Input should be"),
    ("create_support_ticket", {"email": "a@example.com", "category": "other", "summary": "x" * 5000}, "summary:"),
    ("no_such_tool", {}, "Unknown tool"),
])
def test_validate_args_explains_problems(name, args, message):
    with pytest.raises(ToolInputError) as error:
        validate_args(name, args)
    assert message in str(error.value)


def test_tool_schemas_carry_the_constraints():
    schema = get_order_status.to_dict()["input_schema"]
    assert schema["properties"]["order_id"]["pattern"] == "^ORD-\\d{4,8}$"
    assert schema["additionalProperties"] is False


def test_order_totals_add_up():
    for order in ORDERS.values():
        subtotal = round(sum(i["quantity"] * i["unit_price"] for i in order["items"]), 2)
        assert order["subtotal"] == subtotal
        assert order["total"] == round(subtotal + order["shipping_cost"] + order["tax"], 2)


def test_search_store_policies():
    hits = search_store_policies.call({"query": "is shipping free"})
    assert hits[0].chunk.title == "Shipping FAQ > What are the shipping options and costs?"
    assert search_store_policies.call({"query": "what is the capital of France"}) == []


def test_write_tools_dont_run_directly():
    from retailassist.tools import WRITE_TOOLS, create_support_ticket, request_refund
    assert set(WRITE_TOOLS) == {"create_support_ticket", "request_refund"}
    with pytest.raises(RuntimeError, match="approval"):
        create_support_ticket.call({"email": "a@example.com", "category": "other", "summary": "x"})
    with pytest.raises(RuntimeError, match="approval"):
        request_refund.call({"order_id": "ORD-5001", "email": "a@example.com", "reason": "defective"})
