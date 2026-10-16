"""Generate the fictional product catalog and customer orders in data/.

All names, emails, and order details are made up. The random seed is fixed, so
running this again produces the same files.

    python scripts/generate_data.py
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TODAY = date(2026, 10, 15)
random.seed(42)

# (sku, name, category, price, description, final_sale)
CATALOG = [
    ("SKU-1001", "Trailblazer Waterproof Hiking Boots", "footwear", 129.99, "Waterproof leather hiking boots with a grippy rubber sole.", False),
    ("SKU-1002", "CloudStep Running Shoes", "footwear", 89.99, "Lightweight, cushioned running shoes for daily training.", False),
    ("SKU-1003", "Ridgeline Trail Runners", "footwear", 119.00, "Trail running shoes with a rock plate and aggressive lugs.", False),
    ("SKU-1004", "Basecamp Camp Sandals", "footwear", 49.95, "Quick-drying sandals with adjustable straps for camp and water.", False),
    ("SKU-1005", "Frostline Insulated Winter Boots", "footwear", 159.00, "Insulated, waterproof boots rated to -25°F.", False),
    ("SKU-2001", "Summit 30L Daypack", "bags", 64.50, "30-litre backpack with a padded laptop sleeve and rain cover.", False),
    ("SKU-2002", "Expedition 65L Backpacking Pack", "bags", 229.00, "Adjustable-torso pack with a load-lifting hip belt for multi-day trips.", False),
    ("SKU-2003", "Drift 20L Dry Bag", "bags", 29.99, "Roll-top waterproof bag for paddling and wet weather.", False),
    ("SKU-2004", "Trailhead Hip Pack", "bags", 34.00, "Two-litre hip pack with a bottle holster.", False),
    ("SKU-3001", "Merino Wool Hiking Socks (3-pack)", "accessories", 24.00, "Breathable merino socks that resist odour on long hikes.", False),
    ("SKU-3002", "Polarized Trail Sunglasses", "accessories", 59.00, "Lightweight polarized sunglasses with UV400 protection.", False),
    ("SKU-3003", "Packable Sun Hat", "accessories", 27.50, "Wide-brim UPF 50+ hat that folds into a pocket.", False),
    ("SKU-3004", "Trekking Poles (pair)", "accessories", 79.00, "Collapsible aluminium poles with cork grips.", False),
    ("SKU-3005", "Clearance Fleece Gloves", "accessories", 12.00, "Warm fleece gloves with touchscreen fingertips. Final sale.", True),
    ("SKU-4001", "Stormshield Rain Jacket", "apparel", 149.00, "Three-layer waterproof, breathable shell with pit zips.", False),
    ("SKU-4002", "Ember Down Jacket", "apparel", 199.00, "800-fill down jacket that packs into its own pocket.", False),
    ("SKU-4003", "Ridge Softshell Pants", "apparel", 89.00, "Stretchy, water-resistant hiking pants with zip pockets.", False),
    ("SKU-4004", "Trail Tech T-Shirt", "apparel", 32.00, "Moisture-wicking short-sleeve shirt with odour control.", False),
    ("SKU-4005", "Alpine Fleece Pullover", "apparel", 69.00, "Midweight quarter-zip fleece for layering.", False),
    ("SKU-4006", "Clearance Base Layer Top", "apparel", 19.99, "Lightweight synthetic base layer. Final sale.", True),
    ("SKU-5001", "Horizon 2-Person Tent", "camping", 279.00, "Freestanding three-season tent that weighs under 4 lb.", False),
    ("SKU-5002", "Nightfall 20°F Sleeping Bag", "camping", 189.00, "Synthetic mummy bag rated to 20°F with a compression sack.", False),
    ("SKU-5003", "AirRest Sleeping Pad", "camping", 99.00, "Inflatable insulated pad, R-value 4.2.", False),
    ("SKU-5004", "Ultralight Camp Stove", "camping", 54.95, "Compact canister stove that boils a litre in about 3 minutes.", False),
    ("SKU-5005", "Trail Kitchen Cook Set", "camping", 44.00, "Nesting aluminium pot, pan, and two mugs.", False),
    ("SKU-5006", "Lumen 400 Headlamp", "camping", 39.95, "Rechargeable 400-lumen headlamp with red-light mode.", False),
    ("SKU-5007", "CampLite Folding Chair", "camping", 64.00, "Packable chair that holds up to 300 lb.", False),
    ("SKU-6001", "FlowPure Water Filter", "hydration", 36.00, "Squeeze filter that removes 99.9999% of bacteria.", False),
    ("SKU-6002", "Insulated Steel Bottle 32oz", "hydration", 34.95, "Double-wall bottle that keeps drinks cold for 24 hours.", False),
    ("SKU-6003", "Hydration Reservoir 3L", "hydration", 32.00, "Leak-proof bladder with a quick-disconnect hose.", False),
]

FIRST_NAMES = ["Avery", "Jordan", "Priya", "Mateo", "Hana", "Liam", "Sofia", "Kwame", "Elena", "Noah",
               "Aisha", "Ravi", "Chloe", "Diego", "Mei", "Oscar", "Zara", "Tomas", "Ingrid", "Malik",
               "Nora", "Felix", "Leila", "Sam", "Yuki", "Ezra", "Ines", "Caleb", "Amara", "Theo"]
LAST_NAMES = ["Brooks", "Nakamura", "Okafor", "Lindqvist", "Patel", "Moreau", "Castillo", "Hughes", "Adeyemi", "Kowalski",
              "Reyes", "Fitzgerald", "Haddad", "Sorensen", "Tanaka", "Mbeki", "Delgado", "Ferreira", "Novak", "Quinn"]
CITIES = [("Denver", "CO"), ("Portland", "OR"), ("Seattle", "WA"), ("Austin", "TX"), ("Boise", "ID"),
          ("Asheville", "NC"), ("Burlington", "VT"), ("Salt Lake City", "UT"), ("Flagstaff", "AZ"), ("Madison", "WI"),
          ("Missoula", "MT"), ("Bend", "OR"), ("Duluth", "MN"), ("Santa Fe", "NM"), ("Chattanooga", "TN")]
TAX_RATES = {"CO": 0.029, "OR": 0.0, "WA": 0.065, "TX": 0.0625, "ID": 0.06, "NC": 0.0475, "VT": 0.06, "UT": 0.061,
             "AZ": 0.056, "WI": 0.05, "MT": 0.0, "MN": 0.06875, "NM": 0.04875, "TN": 0.07}

# Must match data/policies/shipping-faq.md.
SHIPPING = {
    "standard": {"cost": 6.95, "days": (5, 7), "carriers": ["USPS", "UPS"]},
    "expedited": {"cost": 14.95, "days": (2, 3), "carriers": ["UPS", "FedEx"]},
    "overnight": {"cost": 29.95, "days": (1, 1), "carriers": ["FedEx"]},
}
FREE_SHIPPING_THRESHOLD = 75.00


def add_business_days(start: date, days: int) -> date:
    current = start
    while days > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days -= 1
    return current


def tracking_number(carrier: str) -> str:
    if carrier == "UPS":
        return "1Z" + "".join(random.choices("0123456789ABCDEFGHJKLMNPRSTUVWXYZ", k=16))
    if carrier == "FedEx":
        return "".join(random.choices("0123456789", k=12))
    return "9400" + "".join(random.choices("0123456789", k=18))


def build_products() -> list[dict]:
    products = []
    for sku, name, category, price, description, final_sale in CATALOG:
        stock = random.choice([0, 3, 5, 8, 12, 20, 35, 60, 90])
        products.append({"sku": sku, "name": name, "category": category, "price": price,
                         "stock": stock, "description": description, "final_sale": final_sale})
    return products


def pick_status(age_days: int, method: str) -> str:
    transit = SHIPPING[method]["days"][1]
    if age_days <= 1:
        return "cancelled" if random.random() < 0.05 else "processing"
    if age_days <= transit + 2:
        return random.choice(["shipped", "shipped", "out_for_delivery"])
    roll = random.random()
    if roll < 0.06:
        return "cancelled"
    if roll < 0.14:
        return "return_requested"
    if roll < 0.24:
        return "refunded"
    return "delivered"


def build_orders(products: list[dict]) -> list[dict]:
    customers = [
        {"name": f"{first} {last}", "email": f"{first}.{last}@example.com".lower()}
        for first, last in random.sample([(f, l) for f in FIRST_NAMES for l in LAST_NAMES], 60)
    ]
    orders = []
    for n in range(100):
        # About a third of orders are from the last week, so plenty are still in progress.
        age = random.randint(0, 7) if random.random() < 0.35 else random.randint(8, 75)
        order_date = TODAY - timedelta(days=age)
        customer = random.choice(customers)
        city, state = random.choice(CITIES)
        method = random.choices(["standard", "expedited", "overnight"], weights=[70, 22, 8])[0]

        items = []
        for product in random.sample(products, random.choices([1, 2, 3, 4], weights=[45, 30, 17, 8])[0]):
            qty = random.choices([1, 2, 3], weights=[80, 15, 5])[0]
            items.append({"sku": product["sku"], "name": product["name"], "quantity": qty, "unit_price": product["price"]})

        subtotal = round(sum(i["quantity"] * i["unit_price"] for i in items), 2)
        shipping_cost = 0.0 if method == "standard" and subtotal >= FREE_SHIPPING_THRESHOLD else SHIPPING[method]["cost"]
        tax = round(subtotal * TAX_RATES.get(state, 0.0), 2)
        status = pick_status((TODAY - order_date).days, method)

        order = {
            "order_id": f"ORD-{5001 + n}",
            "order_date": order_date.isoformat(),
            "customer": customer,
            "ship_to": {"city": city, "state": state},
            "items": items,
            "subtotal": subtotal,
            "shipping_method": method,
            "shipping_cost": shipping_cost,
            "tax": tax,
            "total": round(subtotal + shipping_cost + tax, 2),
            "status": status,
            "carrier": None,
            "tracking_number": None,
            "estimated_delivery": None,
            "delivered_date": None,
        }
        if status != "cancelled":
            ship_date = add_business_days(order_date, 1)
            low, high = SHIPPING[method]["days"]
            order["estimated_delivery"] = add_business_days(ship_date, high).isoformat()
            if status != "processing":
                order["carrier"] = random.choice(SHIPPING[method]["carriers"])
                order["tracking_number"] = tracking_number(order["carrier"])
            if status in {"delivered", "return_requested", "refunded"}:
                order["delivered_date"] = add_business_days(ship_date, random.randint(low, high)).isoformat()
        if status == "refunded":
            order["refund_amount"] = round(order["total"] - order["shipping_cost"], 2)
        if status in {"return_requested", "refunded"}:
            order["return_reason"] = random.choice(
                ["Wrong size", "Changed my mind", "Item arrived damaged", "Not as described", "Found a better fit"])
        orders.append(order)

    orders.sort(key=lambda o: o["order_id"])
    return orders


def main() -> None:
    products = build_products()
    orders = build_orders(products)
    (DATA_DIR / "products.json").write_text(json.dumps(products, indent=2, ensure_ascii=False) + "\n")
    (DATA_DIR / "orders.json").write_text(json.dumps(orders, indent=2) + "\n")
    print(f"Wrote {len(products)} products and {len(orders)} orders to {DATA_DIR}")


if __name__ == "__main__":
    main()
