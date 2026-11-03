# Empório da Música: customer service agent

Prototype of a text-message customer service agent for **Empório da Música**, a fictional musical instrument store in Campo Grande/MS. Built for the Artefact AI Engineer technical challenge.

The agent ("Duda") answers customers in Brazilian Portuguese, decides when to query store data (catalog, stock, promotions, orders) and when to read the policy manual (returns, warranty, shipping, payment, hours), computes prices and installments exactly, verifies identity before showing an order, and hands off to a human with a ticket number when it should.

<!-- EXCERPT -->

Full transcripts with the tool calls behind each answer are in [`examples/`](examples/).

---

## Quick start

Requirements: Python 3.10+ and an OpenAI API key.

```bash
git clone https://github.com/Ribeirobusiness/emporio-da-musica-agente.git
cd emporio-da-musica-agente
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                     # then put your key in OPENAI_API_KEY
```

Chat in the terminal:

```bash
emporio-chat                    # new conversation
emporio-chat --debug            # also prints every tool call and token usage
emporio-chat --session ana      # persists the conversation; run again to resume it
```

Other commands:

```bash
pytest                          # 23 deterministic tests, no API key needed
python scripts/data_audit.py    # data-quality findings in the provided dataset
python scripts/run_examples.py  # regenerates the transcripts in examples/ (uses the API)
python scripts/run_examples.py --model gpt-4o-mini --out benchmarks/gpt-4o-mini   # same scenarios, other model
```

Model: `gpt-4.1-mini` by default. Override with `EMPORIO_MODEL` in `.env` or `--model`.

---

## Architecture

```
customer message
      │
      ▼
StoreAgent (agent.py) ── system prompt (prompts.py): persona, rules, policy table of contents
      │   OpenAI Chat Completions with function calling
      │   loop: model → tool_use → Python tool → tool_result → model … (max 8 steps)
      ▼
StoreTools (tools.py) ── deterministic Python, returns JSON
      ├── buscar_produtos / detalhes_produto / listar_promocoes_ativas ─┐
      ├── simular_pagamento                                           ├─ data.py (CSVs → cleaned DataFrames)
      ├── consultar_pedido (order number + e-mail/phone)              ┘
      ├── status_da_loja (Campo Grande clock, hours, holidays)
      ├── consultar_politica ─────────────────────────── policies.py (manual split by section)
      └── encaminhar_para_atendente → logs/encaminhamentos.jsonl (ticket + 24h SLA)
```

| File | Responsibility |
|---|---|
| `src/emporio_agent/data.py` | Loads and cleans the CSVs, joins categories and active promotions, derives `sellable`, data audit |
| `src/emporio_agent/policies.py` | Splits the policy manual into sections and exposes a table of contents |
| `src/emporio_agent/tools.py` | The 8 tools and their JSON schemas |
| `src/emporio_agent/prompts.py` | System prompt (Portuguese, because customers are Brazilian) |
| `src/emporio_agent/agent.py` | Tool-calling loop, history trimming, error fallback. The only provider-specific file |
| `src/emporio_agent/cli.py` | Terminal chat with persisted sessions |
| `data/politicas_da_loja.md` | The PDF manual converted to markdown (tables preserved) |

---

## Technical decisions

### 1. Approach: function calling with a hand-written loop, no framework

The store's questions split cleanly into two kinds: **facts that must be exact** (price, stock, promotion, order status, installment values) and **rules written in prose** (returns, warranty, shipping). The manual itself says the agent must never give prices, stock or deadlines without checking the system (section 7.1), so the core decision was:

> The model decides *what to look up* and *how to say it*. Python computes *everything that must be correct*.

Function calling is the natural fit: each tool is a typed Python function, testable without an LLM, and every decision the agent makes shows up in the trace as a tool call. The loop fits in one short function (`StoreAgent.reply`), which keeps the whole behaviour readable in one file.

Alternatives considered:

- **SQL agent (LLM writes SQL over the CSVs).** Flexible, but the model could write a query that leaks another customer's order, or get a join wrong and quote a wrong price. Typed tools close both doors: order lookup *requires* a matching contact, and prices come from one tested function.
- **LangChain / LangGraph.** Useful when there are many agents, branching graphs or a need to swap providers. Here there is one agent and eight tools; the framework would add abstraction without adding capability. If this grew into several specialised agents (sales, after-sales, logistics) with explicit hand-offs, LangGraph would be my next step.
- **Putting all data in the prompt.** Works for 65 products, breaks at 3,000, and still leaves arithmetic to the model.

### 2. Model and provider: chosen on cost per correct conversation

This agent would run on the store's WhatsApp, at volume, for a small business. So the model choice was treated as a resource decision, measured instead of assumed.

The hard part of this workload is **routing** (which tool, with which arguments, and asking for missing data first), not long reasoning. Arithmetic and rules are already in Python. That points to a small, fast, non-reasoning model with reliable function calling.

I ran the same 5 scripted conversations (`scripts/run_examples.py`) on two candidates and compared tool routing and cost:

<!-- BENCHMARK -->

The provider is isolated in `agent.py`. The project started on the Anthropic API and was moved to OpenAI for cost in a single commit where `agent.py` was the only module whose logic changed (the other files touched were dependencies, the CLI token printout, the loop tests and cost reporting in the example script). That is the practical proof that tools, data, policies and prompt are provider-agnostic. Repeated prefixes (system prompt plus tool schemas, about 3k tokens) are cached automatically by OpenAI, so most input tokens after the first call are billed at the cached rate.

### 3. Retrieval over the policy manual: section lookup, not embeddings

The manual has 10 short sections (about 3k tokens total). I converted the PDF once to markdown, kept the tables, and split it by its own numbered headings. The system prompt carries only a table of contents (section key and what it covers); the model calls `consultar_politica(secao)` and receives the whole section.

Why not vector search: with 10 sections, picking from a table of contents is more reliable than similarity search, needs no embedding model or index, and is deterministic (the same question always returns the same text). Returning *whole* sections also avoids a classic RAG failure in policy documents: retrieving a rule without its exception (for example, the 7-day return window without the list of non-returnable items in 4.4). If the manual grew to hundreds of pages, I would move to hybrid retrieval (BM25 plus embeddings) over sub-sections, with the same tool interface.

### 4. Prompt strategy

- Persona and tone taken from the manual itself (7.1: "informal mas profissional", like a friend who understands music), formatted for WhatsApp (short messages, no tables).
- Explicit "never invent" rules, a "when to use which tool" map, and the manual's special situations (7.3) written as behaviour: out of stock means suggest alternatives, an expired promotion means say so and show the current price, a complaint means ticket plus 24-hour SLA.
- Scope rules: instruments only (no accessories, per section 1), polite refusal for off-topic requests, no legal opinions beyond the manual.
- Basic injection resistance: ignore requests to change the rules, reveal the prompt or grant discounts.
- The current date and time in Campo Grande are injected so the agent can reason about "7 days after receipt" or "are you open now".

### 5. Conversation history

Customers on WhatsApp come back to the same conversation, so history matters. In the CLI, `--session <name>` saves the full message history (including tool calls) as JSON and restores it on the next run. To keep cost bounded, only the last 12 customer turns are sent to the model, and the cut is made only at customer messages so a tool call is never separated from its result. In production this would be keyed by the customer's phone number in a database (Redis or Postgres) instead of a JSON file.

### 6. Data treatment

All cleaning happens once at load time (`data.py`):

- Product `specs` parsed from JSON; categories joined by id.
- Only promotions with `is_active = 1` affect prices. Inactive ones are still reported as "ended" so the agent can tell a customer that a promotion is over instead of ignoring the question.
- `sellable = status == "active" and stock > 0`. Products that are `discontinued` or `coming_soon` are never offered for sale, even when the table shows units in stock (product 130 is `coming_soon` with 2 units).
- Accent-insensitive search text (customers type "violao", "violão" or "violões"); words that name a category become a category filter, so a ukulele whose description mentions "violão" does not show up in a guitar search.
- Phone numbers reduced to digits and e-mails lower-cased for identity checks.

`python scripts/data_audit.py` lists the inconsistencies found in the dataset:

| Finding | How the agent handles it |
|---|---|
| Orders 3 and 20: `total_brl` differs from the sum of their items at list price | The agent reports the order total as recorded and never recalculates past orders (prices may have changed since) |
| Orders 3 and 14 contain product 130, which is `coming_soon` | The agent shows the order as recorded, but will not sell product 130 today |
| Customers 11 and 43 share the same e-mail | Order lookup checks the contact against the customer *of that order*, so a shared e-mail cannot unlock someone else's order |
| Product 96 is active with zero stock; 113 and 136 are discontinued | Excluded from results, listed separately so the agent can suggest alternatives |
| Categories 6, 7 and 8 (wind and orchestral strings) have no products, although the manual says the store sells them | The agent says it has no such item in the catalog right now and offers to forward the request |

Contradictions inside the policy manual, and the reading I assumed:

| Contradiction | Assumption |
|---|---|
| Phone/WhatsApp is (67) 3341-4444 in section 1.2 and (67) 3321-4500 in section 7 | Section 1.2 (company data) is the official contact; if asked about the other number, the agent says the team will confirm |
| E-mail written with an accent in 1.2 (`empóriodamusica`) and without one on the cover | The unaccented address, which is the one that works in practice |
| Payment table says "up to 12x, minimum installment R$ 100"; section 3.1 sets tiers | The detailed tiers in 3.1 prevail: minimum R$ 50 up to 3x ("sem valor mínimo ... exceto abaixo de R$ 50,00"), R$ 80 for 4x to 6x, R$ 100 for 7x to 12x |
| The manual mentions "mais de 300 instrumentos"; the catalog has 65 | The catalog file is the source of truth for what is for sale |

### 7. What the model is not allowed to do

| Not allowed | Enforced by |
|---|---|
| State a price, stock level or delivery estimate without checking | Only tools return them; the prompt forbids answering from memory |
| Do arithmetic (PIX, installments, totals) | `simular_pagamento` computes, the model reads |
| Stack the PIX discount on a promotional price | The tool returns the promo price as the PIX price |
| Offer an expired promotion | Only `is_active = 1` promotions change prices; ended ones are labelled as ended |
| Sell `coming_soon` or discontinued items | `sellable` flag; `simular_pagamento` refuses them |
| Show an order without proof of identity | `consultar_pedido` requires a matching e-mail or phone |
| Read arbitrary data | There is no generic query tool |
| Answer a policy question from memory | Rules come from `consultar_politica`, visible in the trace |

### 8. Order lookup and privacy (LGPD)

`consultar_pedido` needs the order number **and** the e-mail or phone used in the purchase. A wrong contact and a non-existent order return the same message, so the tool cannot be used to discover which order numbers exist. The agent only receives the customer's first name, never the full record.

### 9. Interface

A terminal chat. The challenge asks for the agent working correctly, and a CLI keeps the focus there. The agent is a plain class (`StoreAgent.reply(text) -> Turn`), so exposing it through FastAPI or a WhatsApp webhook is a thin layer on top.

---

## Example conversations

Generated by `scripts/run_examples.py` against the real model. Each file shows the tool calls behind each answer.

| File | Scenario |
|---|---|
| [01_catalogo_violoes_ate_1000](examples/01_catalogo_violoes_ate_1000.md) | Budget search, hides out-of-stock items, PIX and installments |
| [02_preco_promocao_expirada](examples/02_preco_promocao_expirada.md) | Price lookup, customer asks for a promotion that has ended |
| [03_status_pedido_com_verificacao](examples/03_status_pedido_com_verificacao.md) | Asks for contact before showing the order, detects a late delivery, opens a ticket |
| [04_arrependimento_devolucao](examples/04_arrependimento_devolucao.md) | Right of withdrawal (7 days) applied from the manual |
| [05_fora_do_escopo_e_institucional](examples/05_fora_do_escopo_e_institucional.md) | Accessories and off-topic requests redirected; address and hours |

---

## Known limitations and what I would do with more time

- **Evaluation.** The tools have unit tests, but the agent's answers are only checked through the example scenarios. Next step: a small evaluation set (around 50 questions with expected tool calls and facts) run on every change, using an LLM grader for tone and a deterministic checker for prices and policy facts.
- **Holidays.** Only fixed-date national holidays are handled. Movable ones (Carnaval, Good Friday, Corpus Christi) and Campo Grande municipal holidays are not.
- **Shipping cost.** The manual says outside the metro area it depends on ZIP code, weight and dimensions, which the data does not have. The agent explains the rule and the carriers but does not quote a value.
- **Escalation** is written to a local JSONL file. In production it would open a ticket in the store's help desk or CRM and notify the team.
- **Session storage** is a JSON file per conversation; production would use a database keyed by phone number, with retention rules under LGPD.
- **Single provider at a time.** Switching provider means rewriting the loop in `agent.py` (done once already, Anthropic to OpenAI); a production version would add a fallback provider for outages.
- **Past order totals** are shown as recorded even where they disagree with the item prices; a real deployment would flag these to the team.
- **Interface** is a CLI; next would be a WhatsApp webhook (the store's main channel) with out-of-hours auto-replies, as the manual describes.

---

## Use of AI coding assistants

I used **Claude (Anthropic)** as the main coding assistant, in an agentic setup where it could read the files, run code and tests, and commit. The agent itself runs on OpenAI; the coding assistant and the runtime model are separate choices.

How I used it:

1. **Reading the brief and the data.** Before any code, I had it extract the PDF manual and profile every CSV. That pass found the inconsistencies listed above (order totals, `coming_soon` items inside orders, the two phone numbers in the manual), which shaped the design.
2. **Design first, then code.** We settled the architecture before writing code: typed tools for anything exact, section lookup for policies, no framework. The reasoning in this README comes from that discussion.
3. **Small, tested steps.** Each layer (data, policies, tools, agent loop, CLI) was built and committed separately, with tests written alongside. The commit history shows that progression.
4. **Tests as the safety net.** Some first assumptions were wrong and the tests caught them. For example, a search for "violões nylon" returned a ukulele because its description mentions "violão", which led to treating category words as filters. And my expected installment count for one item was off, because the R$ 80 minimum allows 4x, not 3x.
5. **Two independent builds, then a comparison.** I also had a second version built separately (policy manual pasted whole into the prompt, SQLite instead of DataFrames) and compared the two. I kept this one because policy lookups show up in the trace, which makes "did the agent check the rules?" verifiable, and because its tests and data audit caught more of the inconsistencies in the dataset. From the other version I took the "what the model is not allowed to do" table and the idea of measuring cost per conversation instead of estimating it.
6. **Examples generated, not written.** The transcripts in `examples/` and `benchmarks/` come from running the agent against the real model, not from hand-written dialogue.
