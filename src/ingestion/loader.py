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


def _get_fallback_legacy_docs() -> list[DocFile]:
    return [
        DocFile(
            doc_id="legacy_state-and-lifecycle",
            source="legacy",
            title="State and Lifecycle",
            raw_path="data/raw_docs/react-legacy/content/docs/state-and-lifecycle.md",
            text="State is similar to props, but it is private and fully controlled by the component. In class components, initialize state in the constructor with this.state. State updates are scheduled using this.setState and merged shallowly into the existing state object. Lifecycle methods include componentDidMount, componentDidUpdate, and componentWillUnmount.",
        ),
        DocFile(
            doc_id="legacy_components-and-props",
            source="legacy",
            title="Components and Props",
            raw_path="data/raw_docs/react-legacy/content/docs/components-and-props.md",
            text="Components let you split the UI into independent, reusable pieces. Props are read-only. Whether you declare a component as a function or a class, it must never modify its own props. Pure functions do not attempt to change their inputs.",
        ),
        DocFile(
            doc_id="legacy_handling-events",
            source="legacy",
            title="Handling Events",
            raw_path="data/raw_docs/react-legacy/content/docs/handling-events.md",
            text="Handling events with React elements is very similar to handling events on DOM elements. In class components, you often need to bind event handler methods in the constructor or use class fields syntax.",
        ),
    ]


def _get_fallback_current_docs() -> list[DocFile]:
    return [
        DocFile(
            doc_id="current_reference__react__useState",
            source="current",
            title="useState",
            raw_path="data/raw_docs/react-dev-current/src/content/reference/react/useState.md",
            text="useState is a React Hook that lets you add state to a functional component. Calling useState declares a state variable and a setter function: const [state, setState] = useState(initialState). State updates with an updater function setState(prev => prev + 1) guarantee access to the latest pending state. Unlike this.setState in class components, useState does not automatically merge objects; replace or use the spread operator.",
        ),
        DocFile(
            doc_id="current_reference__react__useEffect",
            source="current",
            title="useEffect",
            raw_path="data/raw_docs/react-dev-current/src/content/reference/react/useEffect.md",
            text="useEffect is a React Hook that lets you synchronize a component with an external system. Pass a setup function and a dependency array. The setup function runs after render and after dependencies change. Return a cleanup function from useEffect to avoid memory leaks when the component unmounts or before re-running the effect.",
        ),
        DocFile(
            doc_id="current_reference__react__useRef",
            source="current",
            title="useRef",
            raw_path="data/raw_docs/react-dev-current/src/content/reference/react/useRef.md",
            text="useRef is a React Hook that lets you reference a value that is not needed for rendering. Calling useRef returns a ref object with a mutable .current property: const ref = useRef(initialValue). Changing a ref does not trigger a component re-render, unlike state variables.",
        ),
        DocFile(
            doc_id="current_learn__synchronizing-with-effects",
            source="current",
            title="Synchronizing with Effects",
            raw_path="data/raw_docs/react-dev-current/src/content/learn/synchronizing-with-effects.md",
            text="Effects let you step outside React and synchronize your components with some external system like non-React widgets, network connections, or the browser DOM. Specify dependencies correctly to avoid infinite loops.",
        ),
        DocFile(
            doc_id="current_reference__react__useMemo",
            source="current",
            title="useMemo",
            raw_path="data/raw_docs/react-dev-current/src/content/reference/react/useMemo.md",
            text="useMemo is a React Hook that lets you cache the result of a calculation between re-renders: const cachedValue = useMemo(calculateValue, dependencies). Use it to skip expensive recalculations.",
        ),
        DocFile(
            doc_id="current_reference__react__useCallback",
            source="current",
            title="useCallback",
            raw_path="data/raw_docs/react-dev-current/src/content/reference/react/useCallback.md",
            text="useCallback is a React Hook that lets you cache a function definition between re-renders: const cachedFn = useCallback(fn, dependencies). It prevents unnecessary re-renders of optimized child components.",
        ),
    ]


def load_legacy_docs(repo_root: str) -> list[DocFile]:
    """Load class-components-era docs from reactjs.org checkout."""
    docs_dir = Path(repo_root) / "content" / "docs"
    if not docs_dir.exists():
        print(f"[Notice] Legacy docs not found at '{docs_dir}'. Using built-in seed docs.", file=sys.stderr)
        return _get_fallback_legacy_docs()
        
    files = sorted(docs_dir.glob("*.md"))
    if not files:
        return _get_fallback_legacy_docs()
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
        print(f"[Notice] Current docs not found at '{root}'. Using built-in seed docs.", file=sys.stderr)
        return _get_fallback_current_docs()
        
    dirs = [root / "learn", root / "reference" / "react"]
    files = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.rglob("*.md")))
            
    if not files:
        return _get_fallback_current_docs()
            
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
