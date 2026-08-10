"""
Ground-truth Q&A authoring helper.

Run this locally in your terminal (it's interactive — needs a real stdin).
It solves the bottleneck of writing high-quality ground-truth evaluation Q&A pairs
by allowing authors to search, view, and select real documentation chunks.

Workflow:
  1. Loads both corpora, chunks them with simple_chunk.
  2. Lets you search chunks by keyword (e.g. "useEffect", "lifecycle").
  3. Ranks matches by term frequency so strong matches appear first.
  4. Displays chunk metadata and full chunk text.
  5. Prompts for question, correct answer, and additional gold chunk IDs.
  6. Incremental auto-save to data/ground_truth/qa_pairs.json.

Usage:
  python scripts/generate_qa_key.py            # default target: 50
  python scripts/generate_qa_key.py --target 100
"""

import json
import sys
import os
import re
import argparse
from pathlib import Path

# Ensure stdout uses utf-8 on Windows to support emojis/unicode
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path for clean package imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.loader import load_legacy_docs, load_current_docs
from src.ingestion.chunkers import simple_chunk
from src.config import GROUND_TRUTH_DIR

# ── ANSI colour palette (matches screenshot exactly) ──────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"

# Foreground colours
CYAN    = "\033[36m"
BRIGHT_CYAN = "\033[96m"
GREEN   = "\033[32m"
BRIGHT_GREEN = "\033[92m"
YELLOW  = "\033[33m"
BRIGHT_YELLOW = "\033[93m"
MAGENTA = "\033[35m"
WHITE   = "\033[37m"
BRIGHT_WHITE = "\033[97m"
RED     = "\033[31m"

QA_FILE     = GROUND_TRUTH_DIR / "qa_pairs.json"
LEGACY_REPO = os.getenv("LEGACY_REPO",  "data/raw_docs/react-legacy")
CURRENT_REPO= os.getenv("CURRENT_REPO", "data/raw_docs/react-dev-current")

WIDTH = 80  # Terminal columns (matches screenshot width)


# ── Helpers ───────────────────────────────────────────────────────────────────

def hr(char="─", width=WIDTH):
    return char * width

def col_sep():
    return f"{DIM} | {RESET}"


def build_chunk_index() -> tuple[list[dict], dict[str, int]]:
    legacy  = load_legacy_docs(LEGACY_REPO)
    current = load_current_docs(CURRENT_REPO)
    docs    = legacy + current

    # Hard assertion: zero doc_id collisions
    ids = [d.doc_id for d in docs]
    assert len(docs) == len(set(ids)), \
        f"Collision! {len(docs)} docs but only {len(set(ids))} unique ids."

    records = []
    counts  = {"legacy": 0, "current": 0}
    for doc in docs:
        for i, chunk_text in enumerate(simple_chunk(doc.text, chunk_size=200, overlap=30)):
            records.append({
                "chunk_id":  f"{doc.doc_id}_chunk{i}",
                "doc_title": doc.title,
                "source":    doc.source,
                "text":      chunk_text,
            })
            counts[doc.source] = counts.get(doc.source, 0) + 1
    return records, counts


def search_chunks(records: list[dict], keyword: str) -> list[dict]:
    kl = keyword.lower()
    hits = [r for r in records if kl in r["text"].lower()]
    hits.sort(key=lambda r: r["text"].lower().count(kl), reverse=True)
    return hits


def load_existing_qa_pairs() -> list[dict]:
    if QA_FILE.exists():
        try:
            return json.loads(QA_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save_qa_pairs(pairs: list[dict]) -> None:
    QA_FILE.parent.mkdir(parents=True, exist_ok=True)
    QA_FILE.write_text(json.dumps(pairs, indent=2), encoding="utf-8")


# ── Dashboard header (matches screenshot layout exactly) ──────────────────────

def _strip_ansi(s: str) -> str:
    """Remove ANSI escape codes to get visible character length."""
    return re.sub(r"\033\[[0-9;]*m", "", s)


def _ljust(s: str, width: int) -> str:
    """Left-justify s to visible width, accounting for invisible ANSI codes."""
    visible = len(_strip_ansi(s))
    return s + " " * max(0, width - visible)


def print_header(records, counts, qa_pairs, target=50):
    n_chunks = len(records)
    n_legacy = counts.get("legacy",  0)
    n_curr   = counts.get("current", 0)
    n_saved  = len(qa_pairs)

    try:
        fp = str(QA_FILE.relative_to(BASE_DIR))
    except ValueError:
        fp = str(QA_FILE)

    # Title — centre on visible width
    title = "📚 Ground-truth Q&A Authoring Helper"
    # emoji is 2 wide, compensate
    visible_title_len = len(title) - 1  # emoji takes 2 cols but counts as 1 char
    pad = max(0, (WIDTH - visible_title_len) // 2)
    print(f"\n{BOLD}{BRIGHT_CYAN}{' ' * pad}{title}{RESET}")
    print(f"{DIM}{'Search real documentation chunks and create high-quality Q&A pairs.':^{WIDTH}}{RESET}")
    print(f"{DIM}{hr()}{RESET}")

    # Three-column stats — use _ljust so ANSI codes don't break alignment
    COL1 = 26
    COL2 = 26

    h1 = f"{BOLD}CORPUS STATUS{RESET}"
    h2 = f"{BOLD}PROGRESS{RESET}"
    h3 = f"{BOLD}DATA FILE{RESET}"
    print(f" {_ljust(h1, COL1)}{col_sep()}{_ljust(h2, COL2)}{col_sep()}{h3}")

    r1c1 = f"Chunks loaded: {BRIGHT_GREEN}{n_chunks}{RESET}"
    r1c2 = f"Questions saved: {BRIGHT_YELLOW}{n_saved} / {target}{RESET}"
    r1c3 = f"{BRIGHT_CYAN}{fp}{RESET}"
    print(f" {_ljust(r1c1, COL1)}{col_sep()}{_ljust(r1c2, COL2)}{col_sep()}{r1c3}")

    r2c1 = f"Sources: legacy ({BRIGHT_GREEN}{n_legacy}{RESET}) + current ({BRIGHT_GREEN}{n_curr}{RESET})"
    r2c2 = f"Target: {target} questions"
    r2c3 = f"{DIM}(auto-saved after each question){RESET}"
    print(f" {_ljust(r2c1, COL1)}{col_sep()}{_ljust(r2c2, COL2)}{col_sep()}{r2c3}")

    print(f"{DIM}{hr()}{RESET}\n")


# ── Main interactive session ──────────────────────────────────────────────────

def run_interactive_session():
    parser = argparse.ArgumentParser(description="Ground-truth Q&A Authoring Helper")
    parser.add_argument("--target", type=int, default=None,
                        help="Number of Q&A pairs to collect (default: max(50, current+10))")
    args = parser.parse_args()

    records, counts = build_chunk_index()
    qa_pairs        = load_existing_qa_pairs()
    # Auto-adjust target so existing pairs never block entry into the loop
    target = args.target if args.target is not None else max(50, len(qa_pairs) + 10)

    print_header(records, counts, qa_pairs, target)

    while len(qa_pairs) < target:
        # ── Search prompt ────────────────────────────────────────────────────
        try:
            keyword = input(
                f"{BRIGHT_CYAN}[{len(qa_pairs)}/{target}]{RESET}"
                f" Search keyword (or '{YELLOW}quit{RESET}' to stop): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession interrupted.")
            break

        if keyword.lower() in ("quit", "exit", "q"):
            break
        if not keyword:
            continue

        matches = search_chunks(records, keyword)
        if not matches:
            print(f"  {YELLOW}No chunks matched. Try a different keyword.{RESET}\n")
            continue

        # ── Results list ─────────────────────────────────────────────────────
        display_count = min(5, len(matches))
        print(f"\n{BRIGHT_GREEN}Found {len(matches)} matching chunks. Showing up to {display_count}:{RESET}\n")

        for i in range(display_count):
            m   = matches[i]
            sc  = BRIGHT_GREEN if m["source"] == "current" else BRIGHT_YELLOW
            snippet = m["text"][:140].replace("\n", " ") + "..."
            cid_str = f"{BRIGHT_CYAN}{m['chunk_id']}{RESET}"
            # Title line: [0] (current) useEffect - React          current_18_chunk3
            label = f"{BRIGHT_CYAN}[{i}]{RESET} ({sc}{m['source']}{RESET}) {BOLD}{m['doc_title']}{RESET}"
            # right-align the chunk_id
            label_plain = f"[{i}] ({m['source']}) {m['doc_title']}"
            pad = max(0, WIDTH - len(label_plain) - len(m["chunk_id"]))
            print(f" {label}{' ' * pad}{cid_str}")
            print(f"   {DIM}{snippet}{RESET}\n")

        # ── Chunk selection ──────────────────────────────────────────────────
        try:
            choice = input(
                f"Pick a chunk number to write a question for "
                f"(or Enter to search again): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not choice.isdigit() or int(choice) >= display_count:
            print()
            continue

        chosen = matches[int(choice)]

        # ── Full chunk text panel ────────────────────────────────────────────
        print(f"\n{BRIGHT_GREEN}Full chunk text:{RESET}")
        print(f"{DIM}{hr()}{RESET}")
        print(chosen["text"])
        print(f"{DIM}{hr()}{RESET}")
        meta = (
            f"Source: {chosen['source']}"
            f"  |  Doc Title: {chosen['doc_title']}"
            f"  |  Chunk ID: {chosen['chunk_id']}"
        )
        print(f"{DIM}{meta}{RESET}\n")

        # ── Q&A authoring ────────────────────────────────────────────────────
        try:
            question = input(
                f"{BRIGHT_GREEN}Write your question:{RESET} "
            ).strip()
            if not question:
                print(f"  {YELLOW}Skipped — empty question.{RESET}\n")
                continue

            answer = input(
                f"{BRIGHT_GREEN}Write the correct answer "
                f"{DIM}(in your own words){RESET}"
                f"{BRIGHT_GREEN}:{RESET} "
            ).strip()

            also_correct = input(
                f"{DIM}Any other chunk_ids also correct? "
                f"(comma-separated, or Enter to skip):{RESET} "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            break

        gold_ids = [chosen["chunk_id"]]
        if also_correct:
            gold_ids.extend([x.strip() for x in also_correct.split(",") if x.strip()])

        qa_pairs.append({
            "id":            f"q{len(qa_pairs) + 1:03d}",
            "question":      question,
            "answer":        answer,
            "gold_chunk_ids": gold_ids,
        })
        save_qa_pairs(qa_pairs)

        print(f"\n{BRIGHT_GREEN}✓ Saved. ({len(qa_pairs)}/{target} total){RESET}")
        print(f"Session in progress. Type '{YELLOW}quit{RESET}' at any time to stop and save your progress.\n")

    print(f"\n{BOLD}{BRIGHT_GREEN}Session ended.{RESET} {len(qa_pairs)}/{target} questions saved to {QA_FILE}")


if __name__ == "__main__":
    run_interactive_session()
