```markdown
# LangGraph × Multi-LLM Book Builder

Generate a book from a short problem statement using a graph of LLM steps:
**spec → toc → plan → draft → images → assemble**.

---

## Why this exists

Large language models often return malformed or chatty JSON. This project was designed to be **fault-tolerant**:

- Extracts JSON fences, but also parses the **first JSON value only**.
- Repairs malformed outputs with a conversion prompt.
- Normalizes output before optional schema validation.
- Never hard-fails: TOC has heuristic parsing and local synthesis fallback.
- Checkpoints runs (`spec.json`, `toc.json`, etc.) for `--resume`.

---

## Features

- **Multi-LLM routing**
  - Ollama: `spec`, `toc`, `plan`, `images`
  - Gemini Flash: `draft`, `assemble`
- **Robust JSON handling**
  - Extract fenced JSON, parse only first JSON, repair when needed
- **Normalization**
  - `spec`: coerce audience/goals/constraints
  - `plan`: coerce objectives/key_ideas/image_prompts
- **TOC resilience**
  - Unwrap wrappers (`{"toc": [...]}`)
  - Parse numbered lines
  - Synthesize fallback TOC with rebalanced pages
- **CLI Options**
  - `--resume`, `--dry_run`, `--sample_chapters`, `--max_workers`, `--strict_json`

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U langgraph langchain-core langchain-ollama langchain-google-genai tenacity jsonschema
ollama pull llama3.1:8b
export GOOGLE_API_KEY="YOUR_KEY"
```

---

## Usage

Minimal run:

```bash
python book.py \
  --problem "How to start a bootstrapped SaaS" \
  --pages 200 \
  --model llama3.1:8b
```

Flags:

- `--out ./book_out` output dir (default `./book_out`)
- `--resume` reuse existing outputs
- `--dry_run` skip draft/assemble
- `--sample_chapters 2` draft only N chapters (0 = all)
- `--max_workers 4` concurrent drafting threads
- `--base_url http://127.0.0.1:11434` custom Ollama URL
- `--google_api_key ...` override env var
- `--strict_json` enforce schema strictly

Outputs:

```
spec.json
toc.json
plans.json
drafts.json
image_prompts.json
book.md
```

---

## Architecture

```
LangGraph StateGraph
  spec  ->  toc  ->  plan  ->  draft  ->  images  ->  assemble  -> END
```

- Structured nodes → **Ollama**
- Heavy drafting/assembly → **Gemini Flash**

---

## Challenges and Fixes

1. **Schema mismatches** (`goals_outcomes` vs `goals`)
   - Fix: Normalize first, validate later.
2. **Non-JSON outputs** (markdown, prose)
   - Fix: Extract fences, parse first JSON, repair with conversion prompt.
3. **`JSONDecodeError: Extra data`**
   - Fix: Parse only first JSON value, ignore trailing text.
4. **Ollama client `temperature` error**
   - Fix: Avoid passing `temperature` param.
5. **TOC hard failures**
   - Fix: unwrap → heuristic numbered lines → fallback synthesis.
6. **Resume loading bad files**
   - Fix: Normalize on load; delete broken files to regenerate.

---

## Potential Pitfalls

- Wrong Ollama URL or missing model → pull model with `ollama pull`.
- Missing/invalid `GOOGLE_API_KEY` → set env var or pass CLI.
- Version mismatches between `langchain-ollama` and `ollama` client.
- Very long inputs exceeding context → trim pages/words.
- Too high concurrency → API rate limits.

---

## Troubleshooting

- **Crash at TOC** → delete `toc.json`, rerun.
- **Extra data error** → ensure first-JSON parsing patch is applied.
- **Ollama temperature error** → update client, don’t pass temperature.
- **Gemini auth error** → check `GOOGLE_API_KEY`, confirm `gemini-2.5-flash` model.

---

## Extending

- Export per-chapter markdown
- Add image generation
- Add revise/review nodes
- Integrate retrieval for references/examples

---

## License

MIT

---

## Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain Ollama](https://python.langchain.com/docs/integrations/llms/ollama/)
- [LangChain Google Generative AI](https://python.langchain.com/docs/integrations/llms/google_generative_ai/)
```
