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
retailassist          # chat in the terminal (prints a session ID; resume with --session <id>)
```

## HTTP API

```bash
export RETAILASSIST_STAFF_API_KEY=some-long-secret   # needed for staff ticket endpoints
uvicorn retailassist.api:app --reload
```

Interactive docs are at http://localhost:8000/docs.

| Endpoint | Who | What it does |
|---|---|---|
| `GET /health` | Anyone | Health check, with product and order counts |
| `POST /chat` | Customers | Send `{"message": "..."}` and get the agent's reply plus a `session_id`. Send the `session_id` back to continue the conversation |
| `GET /orders/{id}?email=...` | Customers | Order details. The email must match the order, otherwise 404 |
| `POST /tickets` | Customers | Open a support ticket: `email`, `category`, `message`, and an optional `order_id` |
| `GET /tickets?status=open` | Staff | List tickets, optionally filtered by status |
| `GET /tickets/{id}` | Staff | One ticket |
| `PATCH /tickets/{id}` | Staff | Set `status` to `open`, `in_progress`, or `closed` |

Staff endpoints need an `X-API-Key` header matching `RETAILASSIST_STAFF_API_KEY`. If that variable isn't set, they're disabled.

Ticket categories: `return`, `refund`, `cancellation`, `address_change`, `damaged_item`, `missing_package`, `other`.

Example:

```bash
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "Do you have rain jackets?"}'

curl "localhost:8000/orders/ORD-5003?email=aisha.ferreira@example.com"

curl localhost:8000/tickets -H "X-API-Key: $RETAILASSIST_STAFF_API_KEY"
```

`/chat` replies also include the message's `intent`. Conversations and tickets are saved in SQLite files (`data/conversations.db` and `data/tickets.db` by default), so both survive restarts.

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

The agent is a [LangGraph](https://langchain-ai.github.io/langgraph/) graph (`src/retailassist/graph.py`). Every customer message goes through these nodes:

```
start_turn → classify → retrieve → call_model ⇄ tool_call
                                       ↓
                                    respond
```

| Node | What it does |
|---|---|
| `classify` | Claude labels the message (`product_question`, `order_status`, `policy_question`, `support_request`, or `other`) and pulls out any order number, email, and product keywords |
| `retrieve` | Looks up matching policy sections, products, and the customer's order (no model call) and attaches them to the customer's message |
| `call_model` | Claude writes the answer, and can ask for tools |
| `tool_call` | Runs the tools Claude asked for and sends the results back (up to 8 rounds per message) |
| `respond` | Produces the final reply, including a polite message if Claude declines |

**Saved conversations.** A LangGraph SQLite checkpointer saves each conversation (`data/conversations.db` by default) under its session ID, so conversations survive restarts. Order numbers and emails a customer mentions are remembered for later messages in the same conversation. The conversation history is only ever appended to, never edited.

Claude is called with the Anthropic SDK directly inside the nodes, using Claude Opus 5.5. If a safety classifier declines a request, it's retried automatically on a fallback model Anthropic picks.

**Tools Claude can call:**

| Tool | What it does |
|---|---|
| `search_products` | Keyword search over the catalog, with an optional max price |
| `get_product_details` | Full details, price, and stock for one SKU |
| `get_order_status` | Status, items, totals, tracking, and delivery dates for an order. Needs the order number **and** the customer's email, so customers can't look up each other's orders |
| `get_store_policy` | The full returns & refund policy or shipping FAQ |
| `create_support_ticket` | Opens a ticket for the support team when a customer wants a return, refund, cancellation, or address change. The agent only does this after the customer agrees |

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
├── graph.py   # LangGraph agent: nodes, prompts, retrieval, saved state
├── agent.py   # terminal chat
├── api.py     # FastAPI app
├── tools.py   # tools the agent can call
├── tickets.py # SQLite ticket storage
└── store.py   # loads the data in data/
data/          # fictional catalog, orders, and policies
scripts/       # generate_data.py rebuilds the catalog and orders
tests/         # tests for the graph, tools, data, and API (no API key needed)
```

## Running tests

```bash
pytest
```

## License

[MIT](LICENSE)
