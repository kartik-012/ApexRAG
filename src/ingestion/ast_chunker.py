"""
AST / Markdown and Code-Block Aware Parent-Document Chunker (Solution 1).

Solves the naive sliding-window limitation by:
1. Parsing Markdown section hierarchy (# H1, ## H2, ### H3, #### H4).
2. Protecting fenced code blocks (```jsx ... ```) so code snippets are never sliced across chunk boundaries.
3. Constructing a Parent-Document Hierarchy:
   - Child Chunks (80-150 words): Fine-grained text blocks and code signatures indexed for high-precision retrieval.
   - Parent Chunks (400-1000 words): The complete enclosing section (heading + explanation + full code examples)
     fed into the LLM context window for rich, un-truncated generation.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class HierarchicalChunk:
    child_id: str
    parent_id: str
    child_text: str
    parent_text: str
    metadata: Dict[str, object] = field(default_factory=dict)


_FENCED_CODE_BLOCK_PATTERN = re.compile(r"(```[a-zA-Z0-9_-]*\n[\s\S]*?\n```)")
_HEADER_PATTERN = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def extract_code_blocks_and_mask(text: str) -> Tuple[str, List[str]]:
    """
    Replaces fenced code blocks with placeholders so markdown headers inside code blocks
    are never mistaken for real document section headers.
    """
    code_blocks = []

    def replacer(match):
        code_blocks.append(match.group(1))
        return f"\n\n__CODE_BLOCK_PLACEHOLDER_{len(code_blocks) - 1}__\n\n"

    masked_text = _FENCED_CODE_BLOCK_PATTERN.sub(replacer, text)
    return masked_text, code_blocks


def unmask_code_blocks(text: str, code_blocks: List[str]) -> str:
    """Restores masked code block placeholders back to their original code."""
    for idx, code in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_PLACEHOLDER_{idx}__", code)
    return text


def split_markdown_sections(text: str, default_title: str = "Overview") -> List[Tuple[str, str, int]]:
    """
    Splits markdown into logical sections based on headings (# through ####).
    Returns a list of (section_title, section_body, header_level).
    Code blocks are protected throughout the split.
    """
    masked_text, code_blocks = extract_code_blocks_and_mask(text)

    # Find all header positions
    matches = list(_HEADER_PATTERN.finditer(masked_text))
    if not matches:
        unmasked_body = unmask_code_blocks(masked_text.strip(), code_blocks)
        return [(default_title, unmasked_body, 1)]

    sections = []
    # Content before the very first header
    if matches[0].start() > 0:
        preamble = masked_text[: matches[0].start()].strip()
        if preamble:
            sections.append((default_title, unmask_code_blocks(preamble, code_blocks), 1))

    for i, m in enumerate(matches):
        header_level = len(m.group(1))
        sec_title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(masked_text)
        raw_body = masked_text[start:end].strip()
        unmasked_body = unmask_code_blocks(raw_body, code_blocks)
        sections.append((sec_title, unmasked_body, header_level))

    return sections


def _subdivide_prose_paragraphs(text: str, max_words: int = 140) -> List[str]:
    """
    Subdivides and groups prose paragraphs into coherent child chunks (80-150 words)
    without breaking sentences across chunk boundaries.
    """
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    child_units = []
    curr_group = []
    curr_words = 0

    for p in raw_paras:
        p_words = len(p.split())
        if p_words > max_words:
            if curr_group:
                child_units.append("\n\n".join(curr_group))
                curr_group = []
                curr_words = 0
            # Sentence split for long paragraphs
            sentences = re.split(r"(?<=[.!?])\s+", p)
            s_curr = []
            s_count = 0
            for s in sentences:
                sw = len(s.split())
                if s_count + sw > max_words and s_curr:
                    child_units.append(" ".join(s_curr))
                    s_curr = [s]
                    s_count = sw
                else:
                    s_curr.append(s)
                    s_count += sw
            if s_curr:
                child_units.append(" ".join(s_curr))
        else:
            if curr_words + p_words > max_words and curr_group:
                child_units.append("\n\n".join(curr_group))
                curr_group = [p]
                curr_words = p_words
            else:
                curr_group.append(p)
                curr_words += p_words

    if curr_group:
        child_units.append("\n\n".join(curr_group))

    return child_units


def hierarchical_markdown_chunk(
    doc_id: str,
    title: str,
    text: str,
    source: str = "current",
    max_child_words: int = 140,
) -> List[HierarchicalChunk]:
    """
    Primary AST-aware Hierarchical Chunker (Solution 1).

    Produces:
      - Child Chunks: Precision retrieval units (prose units + isolated atomic code blocks).
      - Parent Chunks: Complete section context containing full headers, code examples,
        and surrounding explanations for high-fidelity LLM generation.
    """
    sections = split_markdown_sections(text, default_title=title)
    chunks: List[HierarchicalChunk] = []

    for sec_idx, (sec_title, sec_body, header_level) in enumerate(sections):
        parent_id = f"{doc_id}__sec{sec_idx}"
        parent_text = f"{'#' * header_level} {sec_title}\n\n{sec_body}".strip()

        # 1. Extract atomic code blocks from this section
        code_blocks = _FENCED_CODE_BLOCK_PATTERN.findall(sec_body)
        # Prose text with code fences temporarily stripped for paragraph chunking
        prose_only = _FENCED_CODE_BLOCK_PATTERN.sub("", sec_body).strip()

        child_counter = 0

        # Child Type A: Atomic Code Blocks (Never sliced!)
        for code in code_blocks:
            # Extract first line of code for quick signature labeling
            first_line = ""
            lines = code.split("\n")
            if len(lines) > 1 and lines[1].strip():
                first_line = lines[1].strip()[:60]

            child_text = f"[{title} > {sec_title}]\nCode Example:\n{code}".strip()
            chunks.append(
                HierarchicalChunk(
                    child_id=f"{parent_id}__code{child_counter}",
                    parent_id=parent_id,
                    child_text=child_text,
                    parent_text=parent_text,
                    metadata={
                        "source": source,
                        "doc_id": doc_id,
                        "title": title,
                        "section_title": sec_title,
                        "header_level": header_level,
                        "is_code": True,
                        "code_signature": first_line,
                        "parent_id": parent_id,
                    },
                )
            )
            child_counter += 1

        # Child Type B: Coherent Prose Paragraphs
        prose_units = _subdivide_prose_paragraphs(prose_only, max_words=max_child_words)
        for p_text in prose_units:
            child_text = f"[{title} > {sec_title}]\n{p_text}".strip()
            chunks.append(
                HierarchicalChunk(
                    child_id=f"{parent_id}__p{child_counter}",
                    parent_id=parent_id,
                    child_text=child_text,
                    parent_text=parent_text,
                    metadata={
                        "source": source,
                        "doc_id": doc_id,
                        "title": title,
                        "section_title": sec_title,
                        "header_level": header_level,
                        "is_code": False,
                        "parent_id": parent_id,
                    },
                )
            )
            child_counter += 1

        # Fallback: if section had neither code nor multi-sentence prose (short summary)
        if child_counter == 0 and sec_body.strip():
            child_text = f"[{title} > {sec_title}]\n{sec_body.strip()}".strip()
            chunks.append(
                HierarchicalChunk(
                    child_id=f"{parent_id}__full0",
                    parent_id=parent_id,
                    child_text=child_text,
                    parent_text=parent_text,
                    metadata={
                        "source": source,
                        "doc_id": doc_id,
                        "title": title,
                        "section_title": sec_title,
                        "header_level": header_level,
                        "is_code": False,
                        "parent_id": parent_id,
                    },
                )
            )

    return chunks


def hierarchical_chunk_corpus(docs) -> Tuple[List[HierarchicalChunk], Dict[str, str]]:
    """
    Transforms a collection of DocFile objects into hierarchical chunks,
    returning:
      1. All child chunks (for vector store + BM25 indexing)
      2. A lookup table mapping parent_id -> full parent section text
    """
    all_chunks = []
    parent_lookup = {}

    for doc in docs:
        doc_chunks = hierarchical_markdown_chunk(
            doc_id=doc.doc_id,
            title=doc.title,
            text=doc.text,
            source=doc.source,
        )
        all_chunks.extend(doc_chunks)
        for c in doc_chunks:
            if c.parent_id not in parent_lookup:
                parent_lookup[c.parent_id] = c.parent_text

    return all_chunks, parent_lookup
