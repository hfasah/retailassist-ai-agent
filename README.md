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
Assistant: Yes! The Trailblazer Waterproof Hiking Boots are $129.99 and we have 14 in stock.
You: Where's my order ORD-5001?
Assistant: It shipped via UPS and should arrive by October 18.
```

## How it works

The agent uses the Anthropic Python SDK's tool runner: Claude decides when to call a tool, the SDK runs the matching Python function, and the result goes back to Claude until it has an answer.

| Tool | What it does |
|---|---|
| `search_products` | Keyword search over the catalog, with an optional max price |
| `get_product_details` | Full details, price, and stock for one SKU |
| `get_order_status` | Status, items, carrier, and ETA for an order |

The store data in `src/retailassist/store.py` is sample data. Swap those dictionaries for calls to your real catalog, inventory, and order systems.

## Project structure

```
src/retailassist/
├── agent.py   # chat loop, system prompt, and Claude API call
├── tools.py   # tools the agent can call
└── store.py   # sample product and order data
tests/         # unit tests for the tools (no API key needed)
```

## Running tests

```bash
pytest
```

## License

[MIT](LICENSE)
