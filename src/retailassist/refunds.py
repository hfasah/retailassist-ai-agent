"""Refund eligibility, following data/policies/refund-policy.md.

This only decides whether a refund request qualifies and estimates the amount. It never moves money:
an eligible request becomes a refund ticket that a staff member approves.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from .store import PRODUCTS
from .tickets import open_ticket, ticket_store

RefundReason = Literal["changed_mind", "wrong_size", "not_as_described", "defective", "damaged_in_shipping",
                       "wrong_item_sent"]

# From the refund policy.
RETURN_WINDOW_DAYS = 30
DEFECT_WINDOW_DAYS = 365
DAMAGE_REPORT_DAYS = 7
RETURN_SHIPPING_FEE = 7.95
# Our fault: free return shipping, original shipping refunded, and final-sale items still qualify.
OUR_FAULT = {"defective", "damaged_in_shipping", "wrong_item_sent"}


@dataclass
class RefundDecision:
    eligible: bool
    reasons: list[str]                      # why, in plain words; also why not, when ineligible
    items: list[dict] = field(default_factory=list)
    estimated_refund: float = 0.0
    return_shipping_fee: float = 0.0
    next_step: str = ""


def check_refund(order: dict, reason: RefundReason, skus: list[str] | None = None,
                 today: date | None = None) -> RefundDecision:
    today = today or date.today()
    status = order["status"]

    if status == "refunded":
        return RefundDecision(False, ["This order has already been refunded."], next_step="No action needed.")
    if status == "cancelled":
        return RefundDecision(False, ["This order was cancelled, and cancelled orders are refunded in full automatically."],
                              next_step="No action needed.")
    if status == "processing":
        return RefundDecision(False, ["This order hasn't shipped yet, so it can be cancelled instead of refunded."],
                              next_step="Open a ticket with category 'cancellation'.")
    if status in {"shipped", "out_for_delivery"}:
        return RefundDecision(False, ["This order hasn't been delivered yet. A return can be started once it arrives."],
                              next_step="Wait for delivery, then request the refund.")

    delivered = date.fromisoformat(order["delivered_date"])
    days_since = (today - delivered).days
    ordered = {item["sku"]: item for item in order["items"]}
    wanted = [s.strip().upper() for s in skus] if skus else list(ordered)
    unknown = [s for s in wanted if s not in ordered]
    if unknown:
        return RefundDecision(False, [f"These items aren't in the order: {', '.join(unknown)}."],
                              next_step="Check the SKUs with lookup_order.")

    reasons: list[str] = []
    if reason == "damaged_in_shipping":
        if days_since > DAMAGE_REPORT_DAYS:
            return RefundDecision(False, [
                f"Damage has to be reported within {DAMAGE_REPORT_DAYS} days of delivery; this order was delivered "
                f"{days_since} days ago. If the item is defective, it may still be covered for a year."])
    elif reason == "defective":
        if days_since > DEFECT_WINDOW_DAYS:
            return RefundDecision(False, [f"Defects are covered for {DEFECT_WINDOW_DAYS} days after delivery; this "
                                          f"order was delivered {days_since} days ago."])
    elif days_since > RETURN_WINDOW_DAYS:
        return RefundDecision(False, [f"Returns are accepted within {RETURN_WINDOW_DAYS} days of delivery; this order "
                                      f"was delivered {days_since} days ago."])

    items, excluded = [], []
    for sku in wanted:
        item = ordered[sku]
        final_sale = PRODUCTS.get(sku, {}).get("final_sale", False)
        if final_sale and reason not in OUR_FAULT:
            excluded.append(item["name"])
            continue
        items.append({"sku": sku, "name": item["name"], "quantity": item["quantity"],
                      "amount": round(item["quantity"] * item["unit_price"], 2)})
    if excluded:
        reasons.append(f"Final sale items can't be returned: {', '.join(excluded)}.")
    if not items:
        return RefundDecision(False, reasons, next_step="No refundable items in this request.")

    # Item price plus the sales tax paid on it (tax is spread across the order by price).
    item_total = sum(i["amount"] for i in items)
    tax = order["tax"] * item_total / order["subtotal"] if order["subtotal"] else 0.0
    refund = item_total + tax
    fee = 0.0
    if reason in OUR_FAULT:
        reasons.append("Return shipping is free because the problem was on our side.")
        if len(items) == len(order["items"]):
            refund += order["shipping_cost"]
            reasons.append("Original shipping is refunded too.")
    else:
        fee = RETURN_SHIPPING_FEE
        refund -= fee
        reasons.append(f"A ${fee:.2f} return shipping fee is deducted.")

    reasons.insert(0, f"Delivered {days_since} days ago, within the return window.")
    return RefundDecision(True, reasons, items, round(max(refund, 0.0), 2), fee,
                          next_step="A team member will review and approve the refund; the customer gets a prepaid "
                                    "return label by email.")


def refund_outcome(order: dict, reason: RefundReason, skus: list[str] | None = None) -> dict:
    """What a refund request for this order would come to, without changing anything.

    Includes any refund ticket already open for the order, so repeat requests don't open another.
    """
    decision = check_refund(order, reason, skus)
    outcome = {"eligible": decision.eligible, "reasons": decision.reasons, "items": decision.items,
               "estimated_refund": decision.estimated_refund, "return_shipping_fee": decision.return_shipping_fee,
               "ticket_id": None, "already_requested": False, "next_step": decision.next_step}
    existing = ticket_store.find_open(order_id=order["order_id"], category="refund")
    if existing:
        outcome.update(ticket_id=existing["ticket_id"], already_requested=True,
                       next_step=f"A refund request is already open ({existing['ticket_id']}); a team member will follow up.")
    return outcome


def needs_ticket(outcome: dict) -> bool:
    return outcome["eligible"] and not outcome["already_requested"]


def describe_refund(reason: str, outcome: dict, details: str = "") -> str:
    lines = [f"Refund request ({reason.replace('_', ' ')}), estimated ${outcome['estimated_refund']:.2f}:"]
    lines += [f"- {i['quantity']} x {i['name']} ({i['sku']}): ${i['amount']:.2f}" for i in outcome["items"]]
    lines += [f"- {r}" for r in outcome["reasons"]]
    if details:
        lines.append(f"Customer notes: {details}")
    return "\n".join(lines)


def submit_refund(order: dict, email: str, reason: RefundReason, skus: list[str] | None = None, details: str = "",
                  *, source: str, note: str = "") -> dict:
    """Check the policy again and, if the request still qualifies, open the refund ticket.

    Only call this once a person has approved the request (see approvals.py).
    """
    outcome = refund_outcome(order, reason, skus)
    if needs_ticket(outcome):
        message = describe_refund(reason, outcome, details) + (f"\n{note}" if note else "")
        ticket = open_ticket(email=email, category="refund", message=message, order_id=order["order_id"],
                             source=source)
        outcome["ticket_id"] = ticket["ticket_id"]
    return outcome
