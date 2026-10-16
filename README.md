# RetailAssist AI Agent

An AI shopping and customer-support agent for retail, built on [Gemini](https://ai.google.dev/) or [Claude](https://www.anthropic.com/claude). It chats with customers, searches the product catalog, checks stock and prices, and looks up order status.

## Getting started

Requires Python 3.11+ and a [Gemini API key](https://aistudio.google.com/apikey) (or an [Anthropic API key](https://console.anthropic.com/) to use Claude).

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
export RETAILASSIST_STAFF_API_KEY=some-long-secret   # needed for staff ticket and approval endpoints
uvicorn retailassist.api:app --reload
```

Interactive docs are at http://localhost:8000/docs.

| Endpoint | Who | What it does |
|---|---|---|
| `GET /health` | Anyone | Health check: provider and model, plus product, order, and indexed policy-section counts |
| `POST /chat` | Customers | Send `{"message": "..."}` and get the agent's reply plus a `session_id`. Send the `session_id` back to continue the conversation |
| `GET /chat/{session_id}` | Customers | The conversation's latest reply, and any requests it's waiting on approval for |
| `GET /orders/{id}?email=...` | Customers | Order details. The email must match the order, otherwise 404; too many misses give 429 |
| `POST /tickets` | Customers | Open a support ticket yourself (a form, not the agent): `email`, `category`, `message`, and an optional `order_id` |
| `GET /approvals?status=pending` | Staff | Write actions waiting for a decision (pending by default; also `approved`, `rejected`) |
| `GET /approvals/{id}` | Staff | One request, with exactly what it will do |
| `POST /approvals/{id}` | Staff | `{"decision": "approve" \| "reject", "note": "..."}`. Approved actions run straight away, and a waiting chat continues; the response includes the customer's new reply |
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

curl localhost:8000/approvals -H "X-API-Key: $RETAILASSIST_STAFF_API_KEY"
curl -X POST localhost:8000/approvals/1 -H "X-API-Key: $RETAILASSIST_STAFF_API_KEY" \
  -H 'content-type: application/json' -d '{"decision": "approve", "note": "Label sent"}'
```

`/chat` replies also include the message's `intent` and a `citations` list (`number`, `title`, `source`, and the exact `quotes` relied on), numbered to match the `[n]` markers in the reply. Conversations, tickets, and approvals are saved in SQLite files (`data/conversations.db`, `data/tickets.db`, and `data/approvals.db` by default), so they survive restarts.

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
start_turn → classify → retrieve → call_model → review_tool_calls → approval_gate → tool_call ─┐
                                       ↑ ↓                                                    │
                                       │ respond                                              │
                                       └──────────────────────────────────────────────────────┘
```

| Node | What it does |
|---|---|
| `classify` | The model labels the message (`product_question`, `order_status`, `policy_question`, `support_request`, or `other`) and pulls out any order number, email, and product keywords |
| `retrieve` | Semantic search over the policy documents, plus matching products and the customer's order (no model call). Policy sections are attached as citable search results |
| `call_model` | The model writes the answer, and can ask for tools |
| `review_tool_calls` | Checks every tool call before anything runs: order access (below), argument validation, and turning write actions into requests for approval |
| `approval_gate` | Pauses the conversation (a LangGraph `interrupt`) until a staff member approves or rejects each write action |
| `tool_call` | Runs the allowed tools, and approved write actions, and sends the results back (up to 8 rounds per message) |
| `respond` | Produces the final reply with numbered `[1]` citation markers and a list of sources, or a polite message if the model declines |

**Policy search with citations.** The policy documents are split into one chunk per section, embedded with a small local model (`all-MiniLM-L6-v2`, about 80 MB, downloaded on first run), and stored in a [Chroma](https://www.trychroma.com/) vector database in `data/vectordb/`. The index rebuilds itself when the policy files change. Search matches by meaning, so "how long until my money comes back" finds the Refunds section with no shared keywords. How citations work depends on the provider (see below). Replies come back like:

```
Refunds go back to your original payment method[1] and usually show up within 5–10 business days.
  [1] Returns & Refund Policy > Refunds (data/policies/refund-policy.md#refunds)
```

**Human approval for write actions.** Nothing the agent does changes data until a person approves it. When the model asks to open a ticket or request a refund, `review_tool_calls` saves the request in `data/approvals.db` and `approval_gate` pauses the graph. The customer gets "Before I go ahead, a team member needs to approve this: …", and `/chat` returns the request under `pending_approvals`. New messages in that conversation get 409 until it's decided. A staff member reviews it with `GET /approvals` and decides with `POST /approvals/{id}`. The graph then resumes from its checkpoint, runs the action only if approved, and the model tells the customer the outcome, which the customer sees with `GET /chat/{session_id}`. A refund that doesn't qualify under the policy skips approval, because nothing would be written. The same queue holds write requests from the MCP server. In the terminal chat, the conversation waits and you can press Enter to check whether it's been decided.

**Validated tool inputs and outputs.** Tool arguments are typed with shared constraints (`src/retailassist/schemas.py`): order numbers look like `ORD-5001`, emails like an email, SKUs like `SKU-1001`, and text fields have length limits. Unknown arguments are rejected. The same types are used by the agent's tools, the MCP server, and the approval queue, and the constraints go into the JSON schemas the models see.
- **Inputs:** before anything runs, `review_tool_calls` validates every call and normalizes it (`" ord-5003"` becomes `ORD-5003`). A bad call never reaches the access checks, staff, or the data. The model gets one line per bad field, such as `order_id: should look like ORD-5001 (got "5003")`, so it can fix the call or ask the customer.
- **Gemini schemas:** Gemini only supports some JSON Schema keywords, so its schemas leave out `pattern` and length limits and describe them in the field's text instead. The server still enforces them.
- **Tool outputs:** tools return output models (`Order`, `Product`, `RefundOutcome`, ...). Building them validates the data, and they keep out what the model doesn't need, such as the customer's stored email.
- **Classification output:** the classifier uses structured output on both providers, and the reply is checked against the `Classification` model. A malformed order number or email is dropped rather than trusted. If the output is unusable, the message is treated as `other` and a warning is logged.

**Order access.** Every order lookup, in the agent, the API, and the MCP server, goes through `src/retailassist/access.py`:
- The order number must match the email it was placed with. A wrong number and a wrong email get the same answer, so nobody can find out which orders exist.
- After 5 misses in 15 minutes, order lookups are paused for that conversation and for that client IP (and for MCP clients as a group). Behind a proxy, run uvicorn with `--proxy-headers` so the limit applies to customers, not the proxy. The counter is in memory, so with several server instances each one keeps its own.
- Once a conversation has verified one email, it can't look up or file requests for orders under another, so one chat can't be used to probe other customers' orders.
- The agent checks access before it runs any tool that takes an order, and before it asks staff to approve anything.

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
| `create_support_ticket` | Asks for a ticket for the support team (a return, cancellation, address change, damaged or missing package). The agent only asks after the customer agrees, and a staff member approves it before it's opened |
| `request_refund` | Checks a refund against the refund policy. If it qualifies, a staff member approves it before the refund ticket is opened; if not, the model gets the reasons |

## MCP server

`retailassist-mcp` exposes order lookup, tickets, and refund requests as [MCP](https://modelcontextprotocol.io/) tools, so any MCP client (Claude Desktop, Claude Code, Gemini CLI, agent frameworks) can use them. It shares the order data, ticket database, and approval queue with the rest of RetailAssist.

| Tool | What it does | Hints |
|---|---|---|
| `lookup_order(order_id, email)` | Status, items, totals, tracking, and delivery dates | read-only, idempotent |
| `create_ticket(email, category, message, order_id?)` | Files a ticket request; a staff member approves it before it's opened | writes |
| `request_refund(order_id, email, reason, skus?, details?)` | Checks the request against the refund policy, estimates the amount, and, if it qualifies, files it for staff approval | writes, idempotent |
| `get_request_status(request_id, email)` | Whether a request is still pending, approved, or rejected, the staff note, and the result (such as the ticket ID) | read-only, idempotent |

- **Customers are verified the same way as everywhere else** (see Order access above), including the lockout after repeated misses.
- **Writes wait for a person.** `create_ticket` and `request_refund` return a `request_id` (`APR-0001`) right away; nothing is written until a staff member approves it with `POST /approvals/{id}`. `get_request_status` only answers for the email the request was made with.
- **`request_refund` never moves money.** It applies the refund policy in code (`src/retailassist/refunds.py`): order status, the 30-day window (7 days for shipping damage, a year for defects), final-sale items, the $7.95 return shipping fee, and tax on the refunded items. Once approved, the request becomes a `refund` ticket with the breakdown. Calling it again for the same order returns the pending request or the open ticket instead of creating a duplicate. Reasons: `changed_mind`, `wrong_size`, `not_as_described`, `defective`, `damaged_in_shipping`, `wrong_item_sent`.
- All tools return structured output with schemas, and errors come back as messages the model can act on (for example, "ask the customer to double-check both").

**Local (stdio).** Add it to Claude Code:

```bash
claude mcp add retailassist -- ~/retailassist-ai-agent/.venv/bin/retailassist-mcp
```

or to Claude Desktop's `claude_desktop_config.json`:

```json
{"mcpServers": {"retailassist": {"command": "/Users/you/retailassist-ai-agent/.venv/bin/retailassist-mcp"}}}
```

**Remote (streamable HTTP).** Every request needs `Authorization: Bearer $RETAILASSIST_MCP_TOKEN`; the server won't start over HTTP without it.

```bash
export RETAILASSIST_MCP_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
retailassist-mcp --transport http --host 0.0.0.0 --port 8080   # serves /mcp; PORT is honoured too
claude mcp add --transport http retailassist https://your-host/mcp --header "Authorization: Bearer $RETAILASSIST_MCP_TOKEN"
```

When bound to `127.0.0.1` (the default), it also only accepts requests addressed to localhost, which blocks DNS rebinding attacks. To explore the tools interactively: `npx @modelcontextprotocol/inspector .venv/bin/retailassist-mcp`.

## Tracing and metrics (Langfuse)

Set your [Langfuse](https://langfuse.com/) keys and every chat turn, approval resume, and MCP tool call is traced:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://us.cloud.langfuse.com   # optional; EU cloud by default, or your self-hosted URL
```

Without the keys, tracing is off and costs nothing. `/health` shows whether it's on, and `/chat` replies include a `trace_id` when it is.

Each turn is one trace, grouped into a Langfuse session by conversation and tagged with the provider and model:

```
chat turn (agent)
├── classify → classify (generation)
├── retrieve (retriever)
├── call_model → call_model (generation)
├── review_tool_calls (guardrail)
├── approval_gate (guardrail)      "Paused for approval", not an error
├── tool_call → get_order_status (tool) ...
└── respond
```

| | What you get |
|---|---|
| **Errors** | An exception marks the step where it happened, and the steps around it, `ERROR` with the message (for example, a Gemini 500 shows on the `classify` generation, its node, and the turn). Invalid tool arguments, refused order lookups, tool errors, refusals, and output that doesn't match the schema are `WARNING`. |
| **Latency** | Every step is timed, so Langfuse shows where a slow turn spent its time. Each turn also logs one line: `chat turn session=… latency_ms=… model_calls=2 tokens_in=… tokens_out=… cost_usd=… tools=… tool_errors=… errors=…`. |
| **Tokens** | Every model call reports input, output, and total tokens, plus cache reads and writes. Gemini's thinking tokens are counted as output, since that's how they're billed. |
| **Cost** | Langfuse prices models it knows. For others, or to use your own rates, set `RETAILASSIST_MODEL_PRICES` to USD per million tokens, e.g. `'{"gemini-3.8-flash": {"input": 0.3, "input_cache_read": 0.03, "output": 2.5}}'`, and each call's cost is sent with it. Each turn's token and cost totals are also attached to its trace. |
| **Privacy** | Email addresses are replaced with `[email]` before anything is sent (`RETAILASSIST_TRACE_MASK_EMAILS=false` to keep them). Verified customer details in the agent's state aren't traced. |

Other settings: `RETAILASSIST_TRACING=off` turns tracing off even with keys set, and `RETAILASSIST_RELEASE` tags traces with a version. Buffered traces are sent when the API, CLI, or MCP server shuts down.

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
├── mcp_server.py # MCP server: lookup_order, create_ticket, request_refund, get_request_status
├── schemas.py # shared input types (OrderId, Email, Sku, ...) and tool output models
├── observability.py # Langfuse tracing, error capture, latency, token and cost metrics
├── access.py  # who may see an order: email match, lockout after misses, one customer per conversation
├── approvals.py # approval queue for write actions, and running them once approved
├── refunds.py # refund eligibility rules from the refund policy
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
