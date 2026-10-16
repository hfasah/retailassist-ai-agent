# RetailAssist AI Agent

An AI shopping and customer-support agent for retail, built on [Gemini](https://ai.google.dev/) or [Claude](https://www.anthropic.com/claude). It chats with customers, searches the product catalog, checks stock and prices, and looks up order status.

## Getting started

Requires Python 3.10+ and a [Gemini API key](https://aistudio.google.com/apikey) (or an [Anthropic API key](https://console.anthropic.com/) to use Claude).

```bash
git clone https://github.com/hfasah/retailassist-ai-agent.git
cd retailassist-ai-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export GEMINI_API_KEY=...             # or put it in .env (see .env.example)
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
| `GET /health` | Anyone | Health check: provider and model, plus product, order, and indexed policy-section counts |
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

`/chat` replies also include the message's `intent` and a `citations` list (`number`, `title`, `source`, and the exact `quotes` relied on), numbered to match the `[n]` markers in the reply. Conversations and tickets are saved in SQLite files (`data/conversations.db` and `data/tickets.db` by default), so both survive restarts.

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
| `classify` | The model labels the message (`product_question`, `order_status`, `policy_question`, `support_request`, or `other`) and pulls out any order number, email, and product keywords |
| `retrieve` | Semantic search over the policy documents, plus matching products and the customer's order (no model call). Policy sections are attached as citable search results |
| `call_model` | The model writes the answer, and can ask for tools |
| `tool_call` | Runs the tools the model asked for and sends the results back (up to 8 rounds per message) |
| `respond` | Produces the final reply with numbered `[1]` citation markers and a list of sources, or a polite message if the model declines |

**Policy search with citations.** The policy documents are split into one chunk per section, embedded with a small local model (`all-MiniLM-L6-v2`, about 80 MB, downloaded on first run), and stored in a [Chroma](https://www.trychroma.com/) vector database in `data/vectordb/`. The index rebuilds itself when the policy files change. Search matches by meaning, so "how long until my money comes back" finds the Refunds section with no shared keywords. How citations work depends on the provider (see below). Replies come back like:

```
Refunds go back to your original payment method[1] and usually show up within 5–10 business days.
  [1] Returns & Refund Policy > Refunds (data/policies/refund-policy.md#refunds)
```

**Saved conversations.** A LangGraph SQLite checkpointer saves each conversation (`data/conversations.db` by default) under its session ID, so conversations survive restarts. Order numbers and emails a customer mentions are remembered for later messages in the same conversation. The conversation history is only ever appended to, never edited.

**Model providers.** The graph, tools, retrieval, and API are shared; only `src/retailassist/llm/` differs per provider. Pick one with `RETAILASSIST_LLM_PROVIDER`:

| | Gemini (default) | Claude |
|---|---|---|
| Model | `gemini-3.8-flash` (`GEMINI_MODEL` to change, e.g. `gemini-4-argon`) | `claude-opus-5-5` (`CLAUDE_MODEL` to change) |
| Credentials | `GEMINI_API_KEY`, or Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI=true` + project and location; on Cloud Run the service account is used) | `ANTHROPIC_API_KEY` |
| Citations | Policy sections are labeled with IDs and Gemini is asked to cite them as `[refund-policy#refunds]`. The server checks every ID against the policy index, drops any that don't exist, and numbers the rest. Sources are section-level, so `quotes` is empty | Sections go in as [search result blocks](https://platform.claude.com/docs/en/build-with-claude/search-results), so the API attaches citations to the exact passages used and `quotes` holds them |
| If the model declines | Polite refusal message | Retried automatically on a fallback model Anthropic picks, then a polite refusal |

Each conversation remembers which provider it started with. Continuing it under the other provider returns 409 (the API) or an error (the CLI), because each provider's saved history includes data only it can read back. Start a new conversation instead.

**Tools the model can call:**

| Tool | What it does |
|---|---|
| `search_products` | Keyword search over the catalog, with an optional max price |
| `get_product_details` | Full details, price, and stock for one SKU |
| `get_order_status` | Status, items, totals, tracking, and delivery dates for an order. Needs the order number **and** the customer's email, so customers can't look up each other's orders |
| `search_store_policies` | Semantic search over the policies, for anything the automatic retrieval missed. Results are citable |
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
├── llm/       # model providers: gemini.py, claude.py, and the shared interface in base.py
├── rag.py     # policy vector index (Chroma) and semantic search
├── agent.py   # terminal chat
├── api.py     # FastAPI app
├── tools.py   # tools the agent can call
├── tickets.py # SQLite ticket storage
└── store.py   # loads the data in data/
data/          # fictional catalog, orders, and policies (plus local databases, not in git)
scripts/       # generate_data.py rebuilds the catalog and orders
tests/         # tests for the graph, tools, data, and API, run against fake Gemini and Claude (no API keys needed)
```

## Running tests

```bash
pytest
```

## License

[MIT](LICENSE)
