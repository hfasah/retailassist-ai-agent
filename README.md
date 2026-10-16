# RetailAssist AI Agent

An AI shopping and customer-support agent for retail, built on [Claude](https://www.anthropic.com/claude). It chats with customers, searches the product catalog, checks stock and prices, and looks up order status.

## Getting started

Requires Python 3.10+ and an [Anthropic API key](https://console.anthropic.com/).

```bash
git clone https://github.com/hfasah/retailassist-ai-agent.git
cd retailassist-ai-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...   # or put it in .env
retailassist
```

Example conversation:

```
You: Do you have hiking boots under $150?
Assistant: Yes! The Trailblazer Waterproof Hiking Boots are $129.99, and we have 3 left in stock.
You: Where's my order ORD-5003? My email is aisha.ferreira@example.com
Assistant: It shipped via UPS and is due to arrive today, October 15.
You: Can I return it if it doesn't fit?
Assistant: Yes. You have 30 days from delivery to return unused items with tags attached...
```

## How it works

The agent uses the Anthropic Python SDK's tool runner: Claude decides when to call a tool, the SDK runs the matching Python function, and the result goes back to Claude until it has an answer.

| Tool | What it does |
|---|---|
| `search_products` | Keyword search over the catalog, with an optional max price |
| `get_product_details` | Full details, price, and stock for one SKU |
| `get_order_status` | Status, items, totals, tracking, and delivery dates for an order. Needs the order number **and** the customer's email, so customers can't look up each other's orders |
| `get_store_policy` | The full returns & refund policy or shipping FAQ |

## Sample data

Everything in `data/` is fictional and made up for testing:

| File | Contents |
|---|---|
| `data/products.json` | 30 outdoor-gear products across 7 categories, with prices, stock levels, and final-sale flags |
| `data/orders.json` | 100 orders (`ORD-5001` to `ORD-5100`) from August to October 2026, with customers, items, shipping, tax, tracking, and statuses from processing to refunded |
| `data/policies/refund-policy.md` | Return window, exclusions, defects, refunds, exchanges, and cancellations |
| `data/policies/shipping-faq.md` | Shipping options and costs, timing, tracking, order statuses, and delivery problems |

Customer emails use the reserved `example.com` domain. The products and orders come from `scripts/generate_data.py`, which uses a fixed random seed, so running it again gives the same files. To connect a real store, replace the loaders in `src/retailassist/store.py` with calls to your catalog, inventory, and order systems.

## Project structure

```
src/retailassist/
├── agent.py   # chat loop, system prompt, and Claude API call
├── tools.py   # tools the agent can call
└── store.py   # loads the data in data/
data/          # fictional catalog, orders, and policies
scripts/       # generate_data.py rebuilds the catalog and orders
tests/         # unit tests for the tools and data (no API key needed)
```

## Running tests

```bash
pytest
```

## License

[MIT](LICENSE)
