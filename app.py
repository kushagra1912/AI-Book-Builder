from __future__ import annotations

import argparse
import json
import os
import re
from json.decoder import JSONDecoder, JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from load_key import load_key

# --- Third-party deps ---
from markdown_pdf import MarkdownPdf, Section
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Optional JSON schema validation
try:
    from jsonschema import validate as _json_validate  # type: ignore
    from jsonschema import ValidationError as _JSONValidationError  # type: ignore
except Exception:
    _json_validate = None

    class _JSONValidationError(Exception): ...


# ---------------------------
# Constants / globals
# ---------------------------
DEFAULT_WORDS_PER_PAGE = 350
DEFAULT_TEMPERATURE = 0.3
STRICT_JSON = False  # toggle via --strict_json


# ---------------------------
# State
# ---------------------------
class BookState(TypedDict, total=False):
    problem: str
    pages_total: int
    words_per_page: int
    spec: Dict[str, Any]
    toc: List[Dict[str, Any]]
    plans: List[Dict[str, Any]]
    drafts: List[Dict[str, Any]]
    image_prompts: List[Dict[str, Any]]
    book_markdown: str
    max_workers: int
    resume: bool
    dry_run: bool


# ---------------------------
# Persistence helpers
# ---------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


# ---------------------------
# JSON helpers
# ---------------------------
def extract_json(text: str) -> str:
    """Extract a JSON object/array from chatty model output (keeps full candidate)."""
    fence = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    i = text.find("{")
    j = text.rfind("}")
    if i != -1 and j != -1 and j > i:
        return text[i : j + 1].strip()
    i = text.find("[")
    j = text.rfind("]")
    if i != -1 and j != -1 and j > i:
        return text[i : j + 1].strip()
    raise ValueError("No JSON found in model output")


def parse_first_json(s: str):
    """
    Return (obj, end_idx) for the FIRST JSON value in s.
    Skips leading noise, ignores trailing text (prevents 'Extra data').
    """
    dec = JSONDecoder()
    i = 0
    while i < len(s) and s[i] not in "{[":
        i += 1
    if i >= len(s):
        raise JSONDecodeError("No JSON start", s, 0)
    obj, end = dec.raw_decode(s, i)
    return obj, end


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_llm_text(llm: Any, system: str, user: str) -> str:
    resp = llm.invoke([("system", system), ("user", user)])
    content = getattr(resp, "content", resp)
    return content if isinstance(content, str) else str(content)


def call_llm_json_lenient(llm: Any, system: str, user: str) -> Any:
    """
    Parse the FIRST JSON value from the model reply, even if there's trailing text.
    Falls back to a conversion prompt if the first parse fails.
    """
    raw = call_llm_text(llm, system, user)
    try:
        candidate = extract_json(raw)
    except Exception:
        candidate = raw
    try:
        obj, _ = parse_first_json(candidate)
        return obj
    except Exception:
        convert_system = (
            "You are a JSON converter. Convert the user's content to STRICT JSON ONLY. "
            "No prose, no code fences."
        )
        repaired = call_llm_text(llm, convert_system, raw)
        try:
            candidate2 = extract_json(repaired)
        except Exception:
            candidate2 = repaired
        obj, _ = parse_first_json(candidate2)
        return obj


def _maybe_validate(payload: Any, schema: Optional[dict]) -> None:
    if STRICT_JSON and _json_validate and isinstance(schema, dict):
        _json_validate(payload, schema)


# ---------------------------
# LLM Router
# ---------------------------
class LLMRouter:
    """Routes tasks to specific LLMs (Ollama vs Gemini)."""

    def __init__(self, ollama: ChatOllama, gemini: ChatGoogleGenerativeAI):
        self.ollama = ollama
        self.gemini = gemini

    def for_node(self, node: str, heavy: bool = False):
        node = node.lower()
        if node in {"spec", "toc", "images"} and not heavy:
            return self.ollama, None  # don't pass temperature to Ollama
        return self.gemini, (0.7 if heavy else 0.6)


def init_llms(
    base_url: Optional[str], model_ollama: str, google_api_key: Optional[str]
) -> LLMRouter:
    if not google_api_key:
        google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError("Google API key required for Gemini")
    # DO NOT set temperature here for ChatOllama (older clients error on it)
    ollama = ChatOllama(model=model_ollama, base_url=base_url)
    gemini = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.6
    )
    return LLMRouter(ollama, gemini)


# ---------------------------
# Spec normalization & schema
# ---------------------------
SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "subtitle": {"type": "string"},
        "audience": {"type": "string"},
        "tone": {"type": "string"},
        "goals": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "audience"],
    "additionalProperties": True,
}


def _coerce_audience(aud: Any) -> str:
    if isinstance(aud, str):
        return aud.strip()
    if isinstance(aud, dict):
        parts = []
        if aud.get("primary"):
            parts.append(f"Primary: {aud['primary']}")
        if aud.get("secondary"):
            parts.append(f"Secondary: {aud['secondary']}")
        return "; ".join(parts) if parts else json.dumps(aud, ensure_ascii=False)
    if isinstance(aud, list):
        return "; ".join(str(x) for x in aud if x)
    return str(aud)


def _coerce_constraints(cons: Any) -> List[str]:
    out: List[str] = []
    if isinstance(cons, list):
        for c in cons:
            if isinstance(c, str):
                s = c.strip()
                if s:
                    out.append(s)
            elif isinstance(c, dict):
                if c.get("constraint"):
                    out.append(str(c["constraint"]).strip())
                else:
                    vals = [str(v).strip() for v in c.values() if v]
                    if vals:
                        out.append(" – ".join(vals))
    elif isinstance(cons, str):
        out = [cons.strip()]
    return [x for x in out if x]


def _coerce_goals(spec: Dict[str, Any]) -> List[str]:
    if isinstance(spec.get("goals"), list) and all(
        isinstance(g, str) for g in spec["goals"]
    ):
        return [g.strip() for g in spec["goals"] if g and isinstance(g, str)]
    if isinstance(spec.get("goals_outcomes"), list):
        goals = []
        for it in spec["goals_outcomes"]:
            if isinstance(it, dict) and it.get("goal"):
                goals.append(str(it["goal"]).strip())
        if goals:
            return goals
    if isinstance(spec.get("objectives"), list):
        vals = [str(x).strip() for x in spec["objectives"] if x]
        if vals:
            return vals
    return []


def normalize_spec(raw_spec: Dict[str, Any]) -> Dict[str, Any]:
    spec: Dict[str, Any] = dict(raw_spec)
    if isinstance(spec.get("title"), str):
        spec["title"] = spec["title"].strip()
    if "audience" in spec:
        spec["audience"] = _coerce_audience(spec["audience"])
    goals = _coerce_goals(spec)
    if goals:
        spec["goals"] = goals
    if "constraints" in spec:
        spec["constraints"] = _coerce_constraints(spec["constraints"])
    if isinstance(spec.get("tone"), str):
        spec["tone"] = spec["tone"].strip()
    if isinstance(spec.get("subtitle"), str):
        spec["subtitle"] = spec["subtitle"].strip()
    spec.setdefault("goals", [])
    return spec


# ---------------------------
# TOC helpers (schema + heuristic + synth)
# ---------------------------
def _toc_schema_obj() -> dict:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "number": {"type": "integer"},
                "title": {"type": "string"},
                "target_pages": {"type": "integer"},
            },
            "required": ["number", "title"],
        },
        "minItems": 6,
    }


def _try_heuristic_toc(lines: str, total_pages: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for ln in lines.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(r"(?i)^(?:chapter\s*)?(\d+)[\.\):\-]*\s+(.*)$", ln)
        if m:
            num = int(m.group(1))
            title = m.group(2).strip(" -:").strip()
            if title:
                items.append({"number": num, "title": title})
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for it in items:
        if it["number"] not in seen:
            seen.add(it["number"])
            uniq.append(it)
    if len(uniq) >= 6:
        for it in uniq:
            it["target_pages"] = 10
        return rebalance_pages(sorted(uniq, key=lambda x: x["number"]), total_pages)
    return []


def _synthesize_toc(
    spec: Dict[str, Any], total_pages: int, chapters: int = 10
) -> List[Dict[str, Any]]:
    """Generate a basic TOC locally to avoid crashing if model refuses structure."""
    title_seed = spec.get("title", "Chapter")
    base = [
        {"number": i + 1, "title": f"{title_seed}: Part {i+1}", "target_pages": 10}
        for i in range(chapters)
    ]
    return rebalance_pages(base, total_pages)


# ---------------------------
# Utility
# ---------------------------
def words_needed(pages: int, wpp: int) -> int:
    return max(100, int(pages * wpp))


def rebalance_pages(
    chapters: List[Dict[str, Any]], total_pages: int
) -> List[Dict[str, Any]]:
    if not chapters:
        return []
    raw = [max(1, int(ch.get("target_pages", 10))) for ch in chapters]
    s = sum(raw)
    if s == total_pages:
        for i, ch in enumerate(chapters):
            ch["target_pages"] = raw[i]
        return chapters
    scaled = [max(1, round(x * total_pages / s)) for x in raw]
    diff = total_pages - sum(scaled)
    i = 0
    while diff != 0 and i < len(scaled) * 2:
        idx = i % len(scaled)
        if diff > 0:
            scaled[idx] += 1
            diff -= 1
        else:
            if scaled[idx] > 1:
                scaled[idx] -= 1
                diff += 1
        i += 1
    for c, p in zip(chapters, scaled):
        c["target_pages"] = p
    return chapters


# ---------------------------
# Node: spec
# ---------------------------
def node_spec(state: BookState, llm: Any) -> BookState:
    system = (
        "Turn a short problem statement into a complete book specification. "
        "Capture title, subtitle, audience, tone, goals/outcomes, constraints."
    )
    user = (
        f"Problem: {state['problem']}\n"
        f"Total pages: {state['pages_total']}\n"
        f"Words per page: {state['words_per_page']}\n"
        "Return ONLY a JSON object. Avoid explanations."
    )
    raw_obj = call_llm_json_lenient(llm, system, user)
    if not isinstance(raw_obj, dict):
        if isinstance(raw_obj, list) and raw_obj and isinstance(raw_obj[0], dict):
            raw_obj = raw_obj[0]
        else:
            raw_obj = {}
    spec = normalize_spec(raw_obj)
    _maybe_validate(spec, SPEC_SCHEMA)
    return {**state, "spec": spec}


# ---------------------------
# Node: toc
# ---------------------------
def node_toc(state: BookState, llm: Any) -> BookState:
    system = (
        "Design a Table of Contents for a practical, non-repetitive book.\n"
        "Return ONLY JSON: an array of objects with fields: number (int), title (string), target_pages (int).\n"
        "No prose, no markdown, no explanations."
    )
    schema = _toc_schema_obj()
    user = (
        f"Book spec: {json.dumps(state['spec'])}\n"
        f"Total pages: {state['pages_total']}\n"
        "Aim for 8–14 chapters. target_pages should roughly sum to the total pages."
    )

    obj = call_llm_json_lenient(llm, system, user)

    # unwrap common wrappers like {"toc":[...]}, {"chapters":[...]}, {"table_of_contents":[...]}
    if isinstance(obj, dict):
        for k in ("toc", "chapters", "table_of_contents", "items"):
            if k in obj and isinstance(obj[k], list):
                obj = obj[k]
                break

    if not isinstance(obj, list):
        # try plain numbered lines → heuristic
        raw_lines = call_llm_text(
            llm, "List the chapters as numbered lines only.", user
        )
        toc = _try_heuristic_toc(raw_lines, state["pages_total"])
        if not toc:
            # last ditch: synthesize locally so we NEVER crash
            toc = _synthesize_toc(
                state.get("spec", {}), state["pages_total"], chapters=10
            )
    else:
        toc = obj

    # coerce & sanitize
    fixed: List[Dict[str, Any]] = []
    for ch in toc:
        if not isinstance(ch, dict):
            continue
        num = ch.get("number")
        title = ch.get("title")
        pages = ch.get("target_pages")
        try:
            num = int(num)
        except Exception:
            continue
        if not isinstance(title, str) or not title.strip():
            continue
        try:
            pages = int(pages) if pages is not None else None
        except Exception:
            pages = None
        fixed.append(
            {
                "number": num,
                "title": title.strip(),
                "target_pages": pages if pages is not None else 10,
            }
        )

    if len(fixed) < 6:
        need = 6 - len(fixed)
        start = max([c["number"] for c in fixed] + [0]) + 1
        for i in range(need):
            fixed.append(
                {"number": start + i, "title": f"Chapter {start+i}", "target_pages": 10}
            )

    fixed = sorted(fixed, key=lambda ch: ch["number"])
    fixed = rebalance_pages(fixed, state["pages_total"])

    for idx, ch in enumerate(fixed, start=1):
        ch["number"] = idx

    _maybe_validate(fixed, schema)
    return {**state, "toc": fixed}


# ---------------------------
# Plan normalization & schema
# ---------------------------
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "number": {"type": "integer"},
        "title": {"type": "string"},
        "objectives": {"type": "array", "items": {"type": "string"}},
        "key_ideas": {"type": "array", "items": {"type": "string"}},
        "image_prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "purpose": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["purpose", "prompt"],
            },
        },
    },
    "required": ["number", "title"],
    "additionalProperties": True,
}


def _as_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, list):
        out: List[str] = []
        for it in x:
            if isinstance(it, str):
                s = it.strip()
                if s:
                    out.append(s)
            elif isinstance(it, dict):
                vals = [str(v).strip() for v in it.values() if v]
                if vals:
                    out.append(" – ".join(vals))
            else:
                s = str(it).strip()
                if s:
                    out.append(s)
        return out
    s = str(x).strip()
    return [s] if s else []


def _normalize_image_prompts(x: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if x is None:
        return out
    if isinstance(x, str):
        s = x.strip()
        if s:
            out.append({"purpose": "illustration", "prompt": s})
        return out
    if isinstance(x, list):
        for it in x:
            if isinstance(it, str):
                s = it.strip()
                if s:
                    out.append({"purpose": "illustration", "prompt": s})
            elif isinstance(it, dict):
                prompt = (
                    it.get("prompt")
                    or it.get("image")
                    or it.get("text")
                    or it.get("caption")
                )
                purpose = it.get("purpose") or it.get("role") or "illustration"
                if prompt:
                    out.append(
                        {"purpose": str(purpose).strip(), "prompt": str(prompt).strip()}
                    )
    elif isinstance(x, dict):
        prompt = x.get("prompt") or x.get("caption") or x.get("text")
        if prompt:
            out.append({"purpose": "illustration", "prompt": str(prompt).strip()})
    return out


def normalize_plan(raw: Dict[str, Any], ch_fallback: Dict[str, Any]) -> Dict[str, Any]:
    plan: Dict[str, Any] = {}
    try:
        plan["number"] = int(raw.get("number", ch_fallback.get("number", 0)))
    except Exception:
        plan["number"] = int(ch_fallback.get("number", 0))
    title = raw.get("title") or ch_fallback.get("title") or f"Chapter {plan['number']}"
    plan["title"] = str(title).strip()
    plan["objectives"] = _as_list_str(
        raw.get("objectives") or raw.get("goals") or raw.get("aims")
    )
    plan["key_ideas"] = _as_list_str(
        raw.get("key_ideas") or raw.get("key points") or raw.get("bullets")
    )
    plan["image_prompts"] = _normalize_image_prompts(
        raw.get("image_prompts") or raw.get("images")
    )
    return plan


# ---------------------------
# Node: plan
# ---------------------------
def node_plan(state: BookState, llm: Any) -> BookState:
    plans: List[Dict[str, Any]] = []
    for ch in state.get("toc", []):
        system = (
            "Create a concrete chapter plan.\n"
            "Return ONLY JSON with fields: number(int), title(str), "
            "objectives(list[str]), key_ideas(list[str]), image_prompts(list[{purpose, prompt}])."
        )
        user = (
            f"Book spec: {json.dumps(state['spec'])}\n"
            f"Chapter: {json.dumps(ch)}\n"
            f"Words per page: {state['words_per_page']}\n"
            "Be specific and avoid repetition."
        )
        raw_plan = call_llm_json_lenient(llm, system, user)
        if not isinstance(raw_plan, dict):
            if (
                isinstance(raw_plan, list)
                and raw_plan
                and isinstance(raw_plan[0], dict)
            ):
                raw_plan = raw_plan[0]
            else:
                raw_plan = {}
        plan = normalize_plan(raw_plan, ch)
        _maybe_validate(plan, PLAN_SCHEMA)
        plans.append(plan)
    return {**state, "plans": plans}


# ---------------------------
# Node: draft
# ---------------------------
def node_draft(
    state: BookState, llm: Any, sample_chapters: Optional[int] = None
) -> BookState:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    drafts: List[Dict[str, Any]] = []
    chapters = state["toc"]
    plans = state["plans"]
    number_to_plan = {int(p.get("number", i + 1)): p for i, p in enumerate(plans)}
    limit = (
        sample_chapters
        if (isinstance(sample_chapters, int) and sample_chapters > 0)
        else len(chapters)
    )
    max_workers = int(state.get("max_workers", 4))

    def draft_one(ch):
        num = int(ch.get("number", 0))
        title = ch.get("title", f"Chapter {num}")
        target_pages = int(ch.get("target_pages", 10))
        target_words = (
            words_needed(target_pages, state["words_per_page"])
            if target_pages > 0
            else 1200
        )
        plan = number_to_plan.get(num, {})
        system = "Draft a coherent, non-repetitive chapter that follows the plan and meets the word budget."
        user = (
            f"Spec: {json.dumps(state['spec'])}\n"
            f"Chapter plan: {json.dumps(plan)}\n"
            f"Target words: {target_words}\n"
            "Write in clear, engaging prose. Do not exceed the target by more than 5%."
        )
        text = call_llm_text(llm, system, user)
        return {"number": num, "title": title, "text": text}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(draft_one, ch): ch for ch in chapters[:limit]}
        for fut in as_completed(futures):
            drafts.append(fut.result())

    drafts.sort(key=lambda d: int(d.get("number", 0)))
    return {**state, "drafts": drafts}


# ---------------------------
# Node: images
# ---------------------------
def node_images(state: BookState, llm: Any) -> BookState:
    prompts: List[Dict[str, Any]] = []
    for p in state.get("plans", []):
        for ip in p.get("image_prompts", []):
            purpose = str(ip.get("purpose", "")).strip()
            prompt = str(ip.get("prompt", "")).strip()
            if purpose and prompt:
                prompts.append(
                    {
                        "chapter": int(p.get("number", 0)),
                        "purpose": purpose,
                        "prompt": prompt,
                    }
                )
    return {**state, "image_prompts": prompts}


# ---------------------------
# Node: assemble
# ---------------------------
def node_assemble(state: BookState, llm: Any, out_dir: Path) -> BookState:
    spec = state.get("spec", {})
    drafts = state.get("drafts", [])

    # Build a unified Markdown string with title, TOC, and chapters
    lines: List[str] = []
    title = spec.get("title", "Book")
    subtitle = spec.get("subtitle")
    lines.append(f"# {title}")
    if subtitle:
        lines.append(f"_{subtitle}_")
    lines.append("\n---\n")
    lines.append("## Table of Contents")
    for d in drafts:
        num = d.get("number", "")
        t = d.get("title", f"Chapter {num}")
        lines.append(f"- [{num}. {t}](#ch{num})")
    lines.append("\n---\n")

    # Append chapters with anchors
    for d in drafts:
        num = d.get("number", "")
        t = d.get("title", f"Chapter {num}")
        lines.append(f'## <a id="ch{num}"></a>{num}. {t}\n')
        lines.append(d.get("text", ""))
        lines.append("\n")

    md_text = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the Markdown for debugging
    (out_dir / "book.md").write_text(md_text, encoding="utf-8")

    # Generate PDF using markdown-pdf
    pdf = MarkdownPdf(toc_level=2)
    # Title page (no TOC entry)
    pdf.add_section(Section(f"# {title}\n_{subtitle}_\n", toc=False))

    # TOC and content as one section
    pdf.add_section(Section(md_text, toc=True))

    pdf_path = out_dir / "book.pdf"
    pdf.meta["title"] = title
    pdf.meta["author"] = "Generated by LLM pipeline"
    pdf.save(str(pdf_path))

    return {**state, "book_markdown": md_text}


# ---------------------------
# Graph
# ---------------------------
def build_graph(
    router: LLMRouter,
    out_dir: Path,
    sample_chapters: Optional[int] = None,
    args: Optional[argparse.Namespace] = None,
):
    builder = StateGraph(BookState)

    def _spec(state):
        path = out_dir / "spec.json"
        if args and getattr(args, "resume", False) and path.exists():
            state["spec"] = _load_json(path)
            return state
        llm, _ = router.for_node("spec")
        new_state = node_spec(state, llm)
        _save_json(path, new_state["spec"])
        return new_state

    def _toc(state):
        path = out_dir / "toc.json"
        if args and getattr(args, "resume", False) and path.exists():
            state["toc"] = _load_json(path)
            return state
        llm, _ = router.for_node("toc")
        new_state = node_toc(state, llm)
        _save_json(path, new_state["toc"])
        return new_state

    def _plan(state):
        path = out_dir / "plans.json"
        if args and getattr(args, "resume", False) and path.exists():
            state["plans"] = _load_json(path)
            return state
        llm, _ = router.for_node("plan")
        new_state = node_plan(state, llm)
        _save_json(path, new_state["plans"])
        return new_state

    def _draft(state):
        if args and getattr(args, "dry_run", False):
            return state
        path = out_dir / "drafts.json"
        if args and getattr(args, "resume", False) and path.exists():
            state["drafts"] = _load_json(path)
            return state
        llm, _ = router.for_node("draft", heavy=True)
        new_state = node_draft(state, llm, sample_chapters=sample_chapters)
        _save_json(path, new_state["drafts"])
        return new_state

    def _images(state):
        if args and getattr(args, "dry_run", False):
            return state
        path = out_dir / "image_prompts.json"
        if args and getattr(args, "resume", False) and path.exists():
            state["image_prompts"] = _load_json(path)
            return state
        llm, _ = router.for_node("images")
        new_state = node_images(state, llm)
        _save_json(path, new_state.get("image_prompts", []))
        return new_state

    def _assemble(state):
        if args and getattr(args, "dry_run", False):
            return state
        path = out_dir / "book.md"
        if args and getattr(args, "resume", False) and path.exists():
            state["book_markdown"] = path.read_text()
            return state
        llm, _ = router.for_node("assemble", heavy=True)
        new_state = node_assemble(state, llm, out_dir=out_dir)
        return new_state

    builder.add_node("spec", _spec)
    builder.add_node("toc", _toc)
    builder.add_node("plan", _plan)
    builder.add_node("draft", _draft)
    builder.add_node("images", _images)
    builder.add_node("assemble", _assemble)

    builder.add_edge("spec", "toc")
    builder.add_edge("toc", "plan")
    builder.add_edge("plan", "draft")
    builder.add_edge("draft", "images")
    builder.add_edge("images", "assemble")
    builder.add_edge("assemble", END)

    builder.set_entry_point("spec")
    return builder.compile()

# out_dir = Path("./books")
# ensure_dir(out_dir)

# router = init_llms(
#     base_url= None,
#     model_ollama= "llama3.1:8b",
#     google_api_key= load_key()
# )

# graph = build_graph(
#     router,
#     out_dir=out_dir,
#     sample_chapters= 2,
# )


# ---------------------------
# CLI / main
# ---------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="LangGraph × Multi-LLM (Ollama + Gemini Flash) — book builder"
    )
    p.add_argument("--problem", required=True, help="Problem to turn into a book")
    p.add_argument("--out", default="./book_out", help="Output directory")
    p.add_argument("--model", default="llama3.1:8b", help="Ollama model name/tag")
    p.add_argument("--pages", type=int, default=200, help="Target total pages")
    p.add_argument(
        "--words_per_page",
        type=int,
        default=DEFAULT_WORDS_PER_PAGE,
        help="Words per page estimate",
    )
    p.add_argument(
        "--base_url",
        default=None,
        help="Override Ollama base URL, e.g., http://0.0.0.0:11434",
    )
    p.add_argument(
        "--google_api_key", default=load_key(), help="Google API key for Gemini"
    )
    p.add_argument(
        "--sample_chapters",
        type=int,
        default=2,
        help="Draft only the first N chapters (0 = all)",
    )
    p.add_argument(
        "--max_workers", type=int, default=4, help="Max threads for concurrent drafting"
    )
    p.add_argument(
        "--resume", action="store_true", help="Reuse existing outputs from the out dir"
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Run spec/toc/plan only; skip drafting and assembly",
    )
    p.add_argument(
        "--strict_json",
        action="store_true",
        help="Enforce JSON schema strictly (off by default)",
    )

    args = p.parse_args(argv)

    global STRICT_JSON
    STRICT_JSON = bool(args.strict_json)

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    router = init_llms(
        base_url=args.base_url,
        model_ollama=args.model,
        google_api_key=args.google_api_key,
    )

    graph = build_graph(
        router,
        out_dir=out_dir,
        sample_chapters=(None if args.sample_chapters == 0 else args.sample_chapters),
        args=args,
    )

    state: BookState = {
        "problem": args.problem,
        "pages_total": args.pages,
        "words_per_page": args.words_per_page,
        "max_workers": int(args.max_workers),
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
    }

    for event in graph.stream(state):
        for _, s in event.items():
            if "spec" in s:
                print("Spec created")
            if "toc" in s:
                print(f"TOC created ({len(s['toc'])} chapters)")
            if "plans" in s:
                print(f"Chapter plans created ({len(s['plans'])})")
            if "drafts" in s:
                print(f"Drafted chapters so far: {len(s['drafts'])}")
            if "image_prompts" in s:
                print(f"Image prompts consolidated: {len(s['image_prompts'])}")
            if "book_markdown" in s:
                print("Book assembled → book.md")

    print("\nOutput folder:", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
