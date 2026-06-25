# LMContextCompiler

A stateful conversation pipeline for local LLMs. Instead of feeding a model an ever-growing transcript, it **externalizes conversation state** to disk and runs each turn as a sequence of small, bounded LLM calls — each one receiving only the state variables it needs, and writing specific results back. The "context" sent to the model is *compiled* from current state every turn rather than accumulated.

> This is a public extraction of a larger private system, mid-refactor. The single-agent social pipeline described below runs end-to-end; some hooks (working memory, multi-agent routing) are present as extension points rather than finished features. See **Scope** at the bottom.

## The idea

A normal chatbot grows its prompt with the whole conversation, which gets slow, expensive, and prone to context rot. This pipeline takes a different approach:

- **State lives outside the model.** Each conversation is a key/value `StateStore` persisted as one JSON file per `chat_id`.
- **Each turn is a small graph of LLM stages**, not one big call. A stage reads a named subset of state, produces a structured (JSON) output, and writes named fields back to state.
- **The model only ever sees compiled, bounded context** — the specific variables a stage declares — so prompt size stays flat no matter how long the conversation runs.

## How a turn works

```
user input
   │
   ▼
 load state from disk  ──►  bootstrap agent / user / scenario (first turn only)
   │
   ▼
 FOREGROUND graph  ──►  returns the user-facing response   ──►  saved to disk
   │
   ▼
 BACKGROUND graphs (in a thread, during the user's typing window)
   │                  update state for the *next* turn
   ▼
 turn gate re-opens
```

The foreground graph is what the user waits on. Background graphs (reflection, memory consolidation, etc.) run off the critical path in a daemon thread, gated so the next turn waits for them to finish. The shipped config uses a single foreground graph; the runner already supports tiered background graphs ordered by dependency.

## The default cognition pipeline

The shipped `main_loop` graph is a four-stage loop, each stage a separate LLM call defined entirely in config (`state_data/configs/main_loop.json`):

1. **input_parser** — reads the user message + world state → writes `input_meaning`, `input_signal`
2. **character_appraisal** — reads the parsed input + persona + trust → writes `internal_reaction`, `engagement_stance`
3. **response_strategy** — reads the appraisal + mood → writes `response_directive`, `response_register`
4. **response_generator** — reads the directive + persona → writes the final `response`

Nothing about these stages is hard-coded — the count, order, prompts, sampling parameters, and which state each reads/writes are all declared in JSON. Changing the pipeline means editing config, not Python.

## Architecture

| Module | Role |
|--------|------|
| `state_manager.py` (`CorePipeline`) | Orchestrator. Loads state, bootstraps agent/user/scenario, runs the LLM stage calls, manages the foreground/background turn gate, persists state. |
| `graph_runner.py` (`GraphRunner`, `GraphConfig`) | Loads `graphs.json`, resolves foreground vs. background graphs, topologically sorts background graphs by `depends_on`, and drives each graph's stages via callbacks. |
| `state_store.py` (`StateStore`) | Dict-backed key/value state with `get` / `set` / `ensure` / `snapshot` / `restore`. Deliberately simple; a vector-store backend can be subbed in by subclassing. |
| `session_store.py` (`SessionStore`) | Per-`chat_id` JSON persistence — one file per conversation. |
| `integrations/ollama_client.py` | Minimal `httpx` client for the Ollama `/api/chat` endpoint, with graceful error returns. |
| `prompt_management/` | `SysPrompt` / `UserPrompt` renderers. User prompts are either loaded from a template file (`{{{variable}}}` substitution) or auto-generated as a labeled JSON block from a stage's `variable_map`. |

## Configuration

State and behavior are data, not code:

- **`state_data/graphs.json`** — defines agent types, their graphs, and tier ordering (tier 0 = foreground, higher tiers = background).
- **`state_data/configs/main_loop.json`** — a graph definition: its stages, each stage's system prompt, sampling params, the state variables it reads (`variable_map`), and the outputs it writes back (`hooks.after.state_updates`).
- **`state_data/agents/<name>.json`** — a character: `name`, `traits`, `style`, `likes`, `dislikes`, `long_term_goals`, etc. These are synthesized into persona prose and an initial emotional state at load time.
- **`state_data/users/<name>.json`** and **`state_data/scenarios/<name>.json`** — the user identity and the world/medium context.
- **`state_data/prompts/system/*.txt`** — the system prompt for each stage.

Templates for agents, users, and scenarios are in `state_data/`.

## Requirements

- Python 3
- [`httpx`](https://www.python-httpx.org/) — `pip install httpx`
- A running [Ollama](https://ollama.com/) server with a chat model pulled

## Running it

1. Start Ollama and pull a model, e.g. `ollama pull llama3`.
2. Create real config files (copy the templates):
   - `state_data/agents/my_agent.json`
   - `state_data/users/my_user.json`
   - `state_data/scenarios/my_scenario.json`
3. Create a test config in `test_configs/` pointing at them:
   ```json
   {
     "agents": ["my_agent"],
     "scenario": "my_scenario",
     "username": "my_user",
     "chat_id": "session_001",
     "first_input": "Hello there"
   }
   ```
4. Point the pipeline at your model and server (the default model name is project-specific, so set this):
   ```bash
   export OLLAMA_MODEL=llama3
   export OLLAMA_URL=http://localhost:11434
   ```
5. Run the interactive harness:
   ```bash
   python testing_pipeline.py --config test_configs/my_config.json
   ```
   Type to chat; `reset` clears the session, `quit` exits. State is written to `./sessions/<chat_id>.json`.

## Scope

This repo is a working skeleton extracted from a larger system, so a few things are intentionally minimal:

- The **single-agent social pipeline** is fully wired and runs end-to-end.
- **Background graphs and the working-memory hook** are supported by the runner but ship empty/as placeholders — they're where the next layer of the system plugs in.
- The `StateStore` is an in-memory dict per turn (persisted as JSON); swapping in a vector or graph backend is an intended extension point.
- Some fields in the agent/scenario templates belong to features of the larger system not yet present in this extraction.
