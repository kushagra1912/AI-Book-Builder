from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from load_key import load_key

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

DEFAULT_WORDS_PER_PAGE = 280  # conservative paperback estimate
DEFAULT_TEMPERATURE = 0.6


# State
class BookState(TypedDict, total=False):
    problem: str
    pages_total: int
    words_per_page: int
    spec: Dict[str, Any]  # title, subtitle, audience, tone, outcomes
    toc: List[Dict[str, Any]]  # [{number, title, summary, target_pages}]
    plans: List[Dict[str, Any]]  # per-chapter plans (sections, objectives, images)
    drafts: List[Dict[str, Any]]  # per-chapter drafts {number, title, markdown}
    image_prompts: List[Dict[str, Any]]  # consolidated prompts
    book_markdown: str


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def extract_json(text: str) -> str:
    """Try to robustly extract a JSON blob from a model response."""
    # Common pattern: model wraps JSON in ```json ... ```
    fence = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    # Fallback: first '{' to last '}'
    i = text.find("{")
    j = text.rfind("}")
    if i != -1 and j != -1 and j > i:
        return text[i : j + 1]
    # As a last resort, return text (may raise on json.loads)
    return text.strip()


def call_llm_json(
    llmOl: ChatOllama, system: str, user: str, schema_hint: str, max_retries: int = 2
) -> Dict[str, Any]:
    """Call the model with instructions to return strict JSON; parse with retries."""
    prompt = (
        f"You are a precise JSON generator. {system}\n\n"
        f"Return ONLY valid JSON matching this schema (no commentary, no code fences).\n\nSchema hint:\n{schema_hint}\n\n"
        f"User request:\n{user}"
    )
    last_err: Optional[Exception] = None
    for _ in range(max_retries + 1):
        resp = llmOl.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        try:
            return json.loads(extract_json(raw))
        except Exception as e:  # noqa: BLE001
            last_err = e
            # Nudge: ask for strict JSON again
            prompt = (
                f"Return ONLY STRICT JSON. Do not include any text outside JSON.\n\n"
                f"User request (repeat):\n{user}\n\nFormat again per schema."
            )
    raise ValueError(f"Failed to parse JSON from model after retries: {last_err}")


def call_llm_text(llmOl: ChatOllama, system: str, user: str) -> str:
    prompt = f"{system}\n\nUser:\n{user}"
    resp = llmOl.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


def rebalance_pages(
    chapters: List[Dict[str, Any]], pages_total: int
) -> List[Dict[str, Any]]:
    """Scale/adjust per‑chapter target_pages so the sum == pages_total (>=1 per chapter)."""
    raw = [max(1, int(c.get("target_pages", 1))) for c in chapters]
    s = sum(raw)
    if s <= 0:
        # evenly distribute
        n = len(chapters)
        base, rem = divmod(pages_total, n)
        for i, c in enumerate(chapters):
            c["target_pages"] = base + (1 if i < rem else 0)
        return chapters

    scale = pages_total / s
    scaled = [max(1, int(round(x * scale))) for x in raw]
    # Fix any rounding drift
    diff = pages_total - sum(scaled)
    i = 0
    while diff != 0 and len(scaled) > 0:
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


def words_needed(pages: int, wpp: int) -> int:
    return max(100, int(pages * wpp))


# Nodes


# Generate Book Specs
def node_spec(state: BookState, llmOl: ChatOllama) -> BookState:
    system = (
        "Turn a short problem statement into a complete book specification. "
        "Capture title, subtitle, audience, tone, goals/outcomes, constraints."
    )
    schema = {
        "title": "string",
        "subtitle": "string",
        "audience": "string",
        "tone": "string",
        "goals": ["string"],
        "constraints": ["string"],
    }
    user = (
        f"Problem: {state['problem']}\n"
        f"Total pages: {state['pages_total']}\n"
        f"Words per page: {state['words_per_page']}"
    )
    spec = call_llm_json(llmOl, system, user, json.dumps(schema, indent=2))
    return {"spec": spec}


# Table of Contents
def node_toc(state: BookState, llmOl: ChatOllama) -> BookState:
    system = (
        "Design a detailed, logical table of contents for a ~200-page book. "
        "Include ~12–18 chapters. For each chapter provide: number, title, summary, target_pages."
    )
    schema = {
        "chapters": [
            {
                "number": "integer",
                "title": "string",
                "summary": "string",
                "target_pages": "integer",
            }
        ]
    }
    user = (
        f"Spec: {json.dumps(state['spec'])}\nPages total: {state['pages_total']}\n"
        "Ensure the sum of target_pages is close to the total."
    )
    toc = call_llm_json(llmOl, system, user, json.dumps(schema, indent=2))
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("Model did not return a valid chapters list")
    chapters = sorted(chapters, key=lambda c: int(c.get("number", 0)))
    chapters = rebalance_pages(chapters, state["pages_total"])  # exact sum
    return {"toc": chapters}


def node_plan(state: BookState, llmOl: ChatOllama) -> BookState:
    plans: List[Dict[str, Any]] = []
    for ch in state["toc"]:
        system = (
            "Create a chapter plan. Include: objectives (3–5), section_outline (5–8 sections\n"
            "with brief bullets), key_concepts, case_studies (optional), image_prompts (0–3)."
        )
        schema = {
            "number": "integer",
            "title": "string",
            "objectives": ["string"],
            "section_outline": [{"heading": "string", "bullets": ["string"]}],
            "key_concepts": ["string"],
            "case_studies": ["string"],
            "image_prompts": [{"purpose": "string", "prompt": "string"}],
        }
        user = (
            f"Book spec: {json.dumps(state['spec'])}\n"
            f"Chapter: {json.dumps(ch)}\n"
            f"Words per page: {state['words_per_page']}\n"
            "Keep the plan concrete and non‑repetitive."
        )
        plan = call_llm_json(llmOl, system, user, json.dumps(schema, indent=2))
        # ensure number/title
        plan["number"] = int(ch.get("number", plan.get("number", 0)))
        plan["title"] = ch.get("title", plan.get("title", "Untitled"))
        plans.append(plan)
    # order by chapter number
    plans.sort(key=lambda p: int(p.get("number", 0)))
    return {"plans": plans}


def node_draft(
    state: BookState, llmOl: ChatOllama, sample_chapters: Optional[int] = None
) -> BookState:
    drafts: List[Dict[str, Any]] = []

    chapters = state["toc"]
    plans = state["plans"]
    number_to_plan = {int(p.get("number", i + 1)): p for i, p in enumerate(plans)}

    limit = (
        sample_chapters
        if (isinstance(sample_chapters, int) and sample_chapters > 0)
        else len(chapters)
    )

    for ch in chapters[:limit]:
        num = int(ch.get("number", 0))
        title = ch.get("title", f"Chapter {num}")
        target_pages = int(ch.get("target_pages", 10))
        target_words = (
            words_needed(target_pages, state["words_per_page"])
            if target_pages > 0
            else 1200
        )

        plan = number_to_plan.get(num, {})
        outline = plan.get("section_outline", [])
        objectives = plan.get("objectives", [])

        system = (
            "Write a polished, well‑structured chapter in Markdown. Use H2 for the chapter title,\n"
            "H3 for sections, short paragraphs, lists, call‑out tips, and concise examples.\n"
            "Avoid fluff and repetition. Maintain consistent tone from the spec."
        )
        user = (
            f"Book spec: {json.dumps(state['spec'])}\n"
            f"Chapter number: {num}\nTitle: {title}\n"
            f"Objectives: {json.dumps(objectives)}\n"
            f"Section outline: {json.dumps(outline)}\n"
            f"Target words (approx): {target_words}.\n"
            "End with a short summary and 3 reflective questions."
        )
        chapter_md = call_llm_text(llmOl, system, user)
        drafts.append({"number": num, "title": title, "markdown": chapter_md})

    # Preserve order across full set (drafts for limited sample will be subset)
    drafts.sort(key=lambda d: int(d.get("number", 0)))
    return {"drafts": drafts}


def node_images(state: BookState) -> BookState:
    # Gather image prompts from plans; de‑duplicate on (chapter, prompt)
    prompts: List[Dict[str, Any]] = []
    for plan, ch in zip(state.get("plans", []), state.get("toc", [])):
        num = int(ch.get("number", plan.get("number", 0)))
        title = ch.get("title", plan.get("title", "Untitled"))
        for item in plan.get("image_prompts", []) or []:
            prompts.append(
                {
                    "chapter": num,
                    "chapter_title": title,
                    "purpose": item.get("purpose", "illustration"),
                    "prompt": item.get("prompt", ""),
                }
            )
    # Simple uniqueness pass
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for p in prompts:
        key = (p["chapter"], p["prompt"].strip())
        if key not in seen and p["prompt"].strip():
            seen.add(key)
            uniq.append(p)
    return {"image_prompts": uniq}


def node_assemble(state: BookState, out_dir: Path) -> BookState:
    spec = state.get("spec", {})
    drafts = state.get("drafts", [])

    parts: List[str] = []
    parts.append(f"# {spec.get('title', 'Untitled Book')}\n")
    if spec.get("subtitle"):
        parts.append(f"*{spec['subtitle']}*\n")
    parts.append("\n")
    parts.append("## About this book\n")
    parts.append(f"**Audience:** {spec.get('audience', 'General readers')}\n\n")
    if spec.get("goals"):
        parts.append("**Goals:**\n")
        for g in spec["goals"]:
            parts.append(f"- {g}")
        parts.append("\n")

    # Table of contents
    parts.append("## Table of Contents\n")
    for ch in state.get("toc", []):
        num = ch.get("number")
        title = ch.get("title")
        pages = ch.get("target_pages")
        parts.append(f"{num}. {title} ({pages} pages)")
    parts.append("\n")

    # Chapters
    ensure_dir(out_dir / "chapters")
    for d in drafts:
        num = int(d.get("number", 0))
        title = d.get("title", f"Chapter {num}")
        md = d.get("markdown", "")
        # add anchor link and write per‑chapter file
        chapter_header = f"\n\n## Chapter {num}: {title}\n\n"
        parts.append(chapter_header)
        parts.append(md.strip())
        with open(
            out_dir / "chapters" / f"chapter_{num:02d}.md", "w", encoding="utf-8"
        ) as f:
            f.write(md)

    book_markdown = "\n".join(parts).strip() + "\n"

    # Write consolidated files
    with open(out_dir / "book.md", "w", encoding="utf-8") as f:
        f.write(book_markdown)

    with open(out_dir / "image_prompts.json", "w", encoding="utf-8") as f:
        json.dump(state.get("image_prompts", []), f, indent=2, ensure_ascii=False)

    with open(out_dir / "toc.json", "w", encoding="utf-8") as f:
        json.dump(state.get("toc", []), f, indent=2, ensure_ascii=False)

    return {"book_markdown": book_markdown}


# Graph


def build_graph(
    llmOl: ChatOllama, out_dir: Path, sample_chapters: Optional[int] = None
):
    builder = StateGraph(BookState)

    # Wrap nodes so they match the (state) -> partial_state signature expected by LangGraph
    def _spec(state: BookState):
        return node_spec(state, llmOl)

    def _toc(state: BookState):
        return node_toc(state, llmOl)

    def _plan(state: BookState):
        return node_plan(state, llmOl)

    def _draft(state: BookState):
        return node_draft(state, llmOl, sample_chapters=sample_chapters)

    def _images(state: BookState):
        return node_images(state)

    def _assemble(state: BookState):
        return node_assemble(state, out_dir)

    builder.add_node("spec", _spec)
    builder.add_node("toc", _toc)
    builder.add_node("plan", _plan)
    builder.add_node("draft", _draft)
    builder.add_node("images", _images)
    builder.add_node("assemble", _assemble)

    builder.set_entry_point("spec")
    builder.add_edge("spec", "toc")
    builder.add_edge("toc", "plan")
    builder.add_edge("plan", "draft")
    builder.add_edge("draft", "images")
    builder.add_edge("images", "assemble")
    builder.add_edge("assemble", END)

    return builder.compile()


# ---------------------------
# CLI / main
# ---------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="LangGraph × Ollama — 200‑page book builder scaffold"
    )
    p.add_argument(
        "--problem", required=True, help="Problem statement to turn into a book"
    )
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
        "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature"
    )
    p.add_argument(
        "--base_url",
        default=None,
        help="Override Ollama base URL, e.g., http://0.0.0.0:11434",
    )
    p.add_argument(
        "--sample_chapters",
        type=int,
        default=2,
        help="Generate only the first N chapters for a quick test (set 0 to draft all)",
    )

    args = p.parse_args(argv)

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # Init local Ollama chat model
    llmOl = ChatOllama(
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
    )

    # Init Gemini LLM
    llmGem = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=load_key())

    # Build graph
    graph = build_graph(
        llmOl,
        out_dir=out_dir,
        sample_chapters=(None if args.sample_chapters == 0 else args.sample_chapters),
    )

    # Initial state
    state: BookState = {
        "problem": args.problem,
        "pages_total": args.pages,
        "words_per_page": args.words_per_page,
    }

    print("\n▶ Running book builder graph...\n")
    # Stream values as each node finishes
    for event in graph.stream(state, stream_mode="values"):
        # Print lightweight progress
        if "spec" in event:
            print("Spec generated")
        if "toc" in event:
            s = (
                sum(int(c.get("target_pages", 0)) for c in event["toc"])
                if isinstance(event["toc"], list)
                else 0
            )
            print(f"TOC created (total pages = {s})")
        if "plans" in event:
            print(f"Chapter plans created ({len(event['plans'])})")
        if "drafts" in event:
            print(f"Drafted chapters so far: {len(event['drafts'])}")
        if "image_prompts" in event:
            print(f"Image prompts consolidated: {len(event['image_prompts'])}")
        if "book_markdown" in event:
            print("Book assembled → book.md")

    print("\n Output folder:", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
