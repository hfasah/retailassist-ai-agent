from datetime import date

import pytest

from retailassist.refunds import RETURN_SHIPPING_FEE, check_refund

TODAY = date(2026, 10, 15)


def make_order(status="delivered", delivered="2026-10-05", items=None, shipping_cost=6.95):
    items = items or [{"sku": "SKU-1001", "name": "Trailblazer Waterproof Hiking Boots", "quantity": 1, "unit_price": 100.0},
                      {"sku": "SKU-3001", "name": "Merino Wool Hiking Socks (3-pack)", "quantity": 2, "unit_price": 25.0}]
    subtotal = sum(i["quantity"] * i["unit_price"] for i in items)
    return {"order_id": "ORD-1", "status": status, "delivered_date": delivered if status in
            {"delivered", "return_requested"} else None, "items": items, "subtotal": subtotal,
            "tax": round(subtotal * 0.1, 2), "shipping_cost": shipping_cost}


@pytest.mark.parametrize("status,fragment", [
    ("refunded", "already been refunded"),
    ("cancelled", "cancelled"),
    ("processing", "can be cancelled"),
    ("shipped", "hasn't been delivered"),
    ("out_for_delivery", "hasn't been delivered"),
])
def test_orders_that_cant_be_refunded_yet(status, fragment):
    decision = check_refund(make_order(status), "changed_mind", today=TODAY)
    assert not decision.eligible and fragment in decision.reasons[0]


def test_eligible_return_deducts_shipping_fee_and_adds_tax():
    decision = check_refund(make_order(), "wrong_size", today=TODAY)
    assert decision.eligible
    # $150 of items + $15 tax - $7.95 return shipping
    assert decision.estimated_refund == round(150 + 15 - RETURN_SHIPPING_FEE, 2)
    assert decision.return_shipping_fee == RETURN_SHIPPING_FEE


def test_partial_refund_gets_its_share_of_tax():
    decision = check_refund(make_order(), "changed_mind", skus=["sku-1001"], today=TODAY)
    assert [i["sku"] for i in decision.items] == ["SKU-1001"]
    assert decision.estimated_refund == round(100 + 10 - RETURN_SHIPPING_FEE, 2)


def test_our_fault_refunds_shipping_and_waives_fee():
    decision = check_refund(make_order(), "damaged_in_shipping", today=date(2026, 10, 9))
    assert decision.eligible
    assert decision.return_shipping_fee == 0
    assert decision.estimated_refund == round(150 + 15 + 6.95, 2)


def test_return_window():
    assert check_refund(make_order(delivered="2026-09-15"), "changed_mind", today=TODAY).eligible      # 30 days
    late = check_refund(make_order(delivered="2026-09-14"), "changed_mind", today=TODAY)              # 31 days
    assert not late.eligible and "30 days" in late.reasons[0]


def test_defects_are_covered_for_a_year_but_damage_must_be_reported_in_a_week():
    old = make_order(delivered="2026-03-01")
    assert check_refund(old, "defective", today=TODAY).eligible
    assert not check_refund(make_order(delivered="2026-10-01"), "damaged_in_shipping", today=TODAY).eligible


def test_final_sale_items():
    items = [{"sku": "SKU-3005", "name": "Clearance Fleece Gloves", "quantity": 1, "unit_price": 12.0}]
    changed_mind = check_refund(make_order(items=items), "changed_mind", today=TODAY)
    assert not changed_mind.eligible and "Final sale" in changed_mind.reasons[0]
    assert check_refund(make_order(items=items), "defective", today=TODAY).eligible


def test_unknown_skus():
    decision = check_refund(make_order(), "changed_mind", skus=["SKU-9999"], today=TODAY)
    assert not decision.eligible and "SKU-9999" in decision.reasons[0]
