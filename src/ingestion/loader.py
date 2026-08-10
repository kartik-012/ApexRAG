"""
Loader for the React documentation corpus.

Two verified sources:
  - LEGACY (class-components era): reactjs/reactjs.org, branch v17.0.2
  - CURRENT (hooks era): reactjs/react.dev, branch main

Includes clean MDX parsing and frontmatter extraction.
"""

import re
import yaml
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DocFile:
    doc_id: str              # stable id, e.g. "legacy_state-and-lifecycle"
    source: str               # "legacy" or "current"
    title: str
    raw_path: str
    text: str                 # cleaned markdown, MDX tags stripped
    code_blocks: list = field(default_factory=list)


MDX_COMPONENT_PATTERN = re.compile(
    r"</?(?:Intro|InlineToc|Pitfall|DeepDive|Note|Sandpack|TeamMember|YouWillLearn|Recap)[^>]*>"
)
HEADING_ANCHOR_PATTERN = re.compile(r"\s*\{/\*[^*]*\*/\}")  # react.dev's {/*anchor-id*/} suffix


def _parse_frontmatter(raw_text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from markdown body."""
    if not raw_text.startswith("---"):
        return {}, raw_text
    parts = raw_text.split("---", 2)
    if len(parts) < 3:
        return {}, raw_text
    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        frontmatter = {}
    body = parts[2].strip()
    return frontmatter, body


def _clean_mdx(body: str) -> str:
    """Strip MDX component wrapper tags, keep inner content; strip heading anchors."""
    body = MDX_COMPONENT_PATTERN.sub("", body)
    body = HEADING_ANCHOR_PATTERN.sub("", body)
    return body.strip()


def load_legacy_docs(repo_root: str) -> list[DocFile]:
    """Load class-components-era docs from reactjs.org checkout."""
    docs_dir = Path(repo_root) / "content" / "docs"
    if not docs_dir.exists():
        print(f"Error: Legacy docs not found at '{docs_dir}'", file=sys.stderr)
        print("Please clone the real repo locally:\n  git clone --depth 1 --branch v17.0.2 https://github.com/reactjs/reactjs.org.git data/raw_docs/react-legacy", file=sys.stderr)
        sys.exit(1)
        
    files = sorted(docs_dir.glob("*.md"))
    result = []
    for f in files:
        raw = f.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(raw)
        
        # Derived from path relative to content root to prevent collisions
        rel_path = f.relative_to(Path(repo_root) / "content")
        doc_id = "legacy_" + str(rel_path.with_suffix('')).replace(os.sep, "__").replace("/", "__")
        
        title = fm.get("title", f.stem)
        result.append(DocFile(
            doc_id=doc_id,
            source="legacy",
            title=title,
            raw_path=str(f),
            text=_clean_mdx(body),
        ))
    return result


def load_current_docs(repo_root: str) -> list[DocFile]:
    """Load hooks-era docs from react.dev checkout."""
    root = Path(repo_root) / "src" / "content"
    if not root.exists():
        print(f"Error: Current docs not found at '{root}'", file=sys.stderr)
        print("Please clone the real repo locally:\n  git clone --depth 1 https://github.com/reactjs/react.dev.git data/raw_docs/react-dev-current", file=sys.stderr)
        sys.exit(1)
        
    dirs = [root / "learn", root / "reference" / "react"]
    files = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.rglob("*.md")))
            
    result = []
    for f in files:
        raw = f.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(raw)
        
        # Derived from path relative to content root to prevent collisions
        rel_path = f.relative_to(root)
        doc_id = "current_" + str(rel_path.with_suffix('')).replace(os.sep, "__").replace("/", "__")
        
        title = fm.get("title", f.stem)
        result.append(DocFile(
            doc_id=doc_id,
            source="current",
            title=title,
            raw_path=str(f),
            text=_clean_mdx(body),
        ))
    return result
