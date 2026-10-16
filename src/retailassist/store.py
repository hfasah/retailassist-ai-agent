"""Sample store data. Replace with calls to your real catalog and order systems."""

PRODUCTS = {
    "SKU-1001": {
        "name": "Trailblazer Waterproof Hiking Boots",
        "category": "footwear",
        "price": 129.99,
        "stock": 14,
        "description": "Waterproof leather hiking boots with a grippy rubber sole.",
    },
    "SKU-1002": {
        "name": "CloudStep Running Shoes",
        "category": "footwear",
        "price": 89.99,
        "stock": 0,
        "description": "Lightweight, cushioned running shoes for daily training.",
    },
    "SKU-2001": {
        "name": "Summit 30L Daypack",
        "category": "bags",
        "price": 64.50,
        "stock": 7,
        "description": "30-litre backpack with a padded laptop sleeve and rain cover.",
    },
    "SKU-3001": {
        "name": "Merino Wool Hiking Socks (3-pack)",
        "category": "accessories",
        "price": 24.00,
        "stock": 52,
        "description": "Breathable merino socks that resist odour on long hikes.",
    },
}

ORDERS = {
    "ORD-5001": {"status": "shipped", "items": ["SKU-1001"], "eta": "2026-10-18", "carrier": "UPS"},
    "ORD-5002": {"status": "processing", "items": ["SKU-2001", "SKU-3001"], "eta": "2026-10-21", "carrier": None},
}
