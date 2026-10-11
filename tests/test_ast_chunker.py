"""
Unit tests for Solution 1: AST / Markdown and Code-Block Aware Parent-Document Chunker.
"""

from src.ingestion.ast_chunker import (
    extract_code_blocks_and_mask,
    unmask_code_blocks,
    split_markdown_sections,
    hierarchical_markdown_chunk,
    hierarchical_chunk_corpus,
    HierarchicalChunk,
)
from src.ingestion.loader import DocFile


SAMPLE_MARKDOWN = """# useEffect

`useEffect` is a React Hook that lets you synchronize a component with an external system.

```jsx
useEffect(setup, dependencies?)
```

## Usage

### Connecting to an external system

Call `useEffect` at the top level of your component to declare an Effect:

```jsx
import { useEffect } from 'react';
import { createConnection } from './chat.js';

function ChatRoom({ roomId }) {
  const [serverUrl, setServerUrl] = useState('https://localhost:1234');

  useEffect(() => {
    const connection = createConnection(serverUrl, roomId);
    connection.connect();
    return () => {
      connection.disconnect();
    };
  }, [serverUrl, roomId]);

  return <h1>Welcome to the {roomId} room!</h1>;
}
```

Some components need to synchronize with external systems like non-React widgets or network APIs.
"""


def test_code_blocks_kept_intact():
    """Fenced code blocks must be completely preserved as atomic units without slicing."""
    chunks = hierarchical_markdown_chunk(
        doc_id="test_useEffect",
        title="useEffect",
        text=SAMPLE_MARKDOWN,
        source="current",
    )
    code_chunks = [c for c in chunks if c.metadata.get("is_code")]
    assert len(code_chunks) >= 2

    # Check the chat room component is in a single code chunk
    chat_chunk = next((c for c in code_chunks if "createConnection" in c.child_text), None)
    assert chat_chunk is not None
    # Code block must start with ```jsx and end with ```
    assert "```jsx" in chat_chunk.child_text
    assert "return <h1>Welcome to the {roomId} room!</h1>;" in chat_chunk.child_text
    assert "```" in chat_chunk.child_text


def test_headers_inside_code_blocks_ignored():
    """Markdown headers (# Header) inside code blocks must NOT split the document section."""
    md_with_comment_header = """## Shell Setup

Run this script in terminal:

```bash
# This is a comment inside bash
curl -s https://example.com | bash
# Another comment
```

This text is still part of the Shell Setup section.
"""
    sections = split_markdown_sections(md_with_comment_header)
    assert len(sections) == 1
    assert sections[0][0] == "Shell Setup"
    assert "curl -s" in sections[0][1]


def test_parent_document_relationship():
    """Every child chunk must link to its parent section containing the full context."""
    chunks = hierarchical_markdown_chunk(
        doc_id="test_useEffect",
        title="useEffect",
        text=SAMPLE_MARKDOWN,
        source="current",
    )
    assert len(chunks) > 0
    for c in chunks:
        assert isinstance(c, HierarchicalChunk)
        assert c.parent_id.startswith("test_useEffect__sec")
        # Parent text must contain header
        assert "#" in c.parent_text
        assert len(c.parent_text) >= len(c.child_text.replace(f"[{c.metadata.get('title')} > {c.metadata.get('section_title')}]\n", ""))


def test_code_vs_prose_labeling():
    """Metadata must accurately distinguish between code snippets and prose text."""
    chunks = hierarchical_markdown_chunk(
        doc_id="test_doc",
        title="Doc",
        text=SAMPLE_MARKDOWN,
        source="current",
    )
    is_code_values = {c.metadata.get("is_code") for c in chunks}
    assert True in is_code_values
    assert False in is_code_values

    for c in chunks:
        if c.metadata.get("is_code"):
            assert "Code Example:\n```" in c.child_text or "```" in c.child_text
        else:
            assert c.metadata.get("is_code") is False


def test_hierarchical_chunk_corpus():
    """Corpus helper must generate unique child chunk IDs and a valid parent lookup map."""
    docs = [
        DocFile(
            doc_id="doc_1",
            source="current",
            title="Hook 1",
            raw_path="doc1.md",
            text="# Hook 1\n\nExplanation.\n\n```js\nconst x = 1;\n```",
        ),
        DocFile(
            doc_id="doc_2",
            source="legacy",
            title="Lifecycle 2",
            raw_path="doc2.md",
            text="# Lifecycle 2\n\nLegacy lifecycle explanation.",
        ),
    ]
    chunks, parent_lookup = hierarchical_chunk_corpus(docs)
    assert len(chunks) >= 2
    assert len(parent_lookup) >= 2

    child_ids = [c.child_id for c in chunks]
    assert len(child_ids) == len(set(child_ids)), "Child chunk IDs must be strictly unique!"

    for c in chunks:
        assert c.parent_id in parent_lookup
        assert parent_lookup[c.parent_id] == c.parent_text
