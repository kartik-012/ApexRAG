# Verified Data Sources

## Legacy (class-components era)
- Repo: https://github.com/reactjs/reactjs.org
- Branch: `v17.0.2`  (⚠️ NOT `main` — main now just redirects to react.dev)
- Path: `content/docs/*.md`
- Files: 91 markdown docs
- Clone command:
  ```
  git clone --depth 1 --branch v17.0.2 https://github.com/reactjs/reactjs.org.git react-legacy
  ```

## Current (hooks era)
- Repo: https://github.com/reactjs/react.dev
- Branch: `main`
- Path: `src/content/learn/*.md` + `src/content/reference/react/*.md`
- Files: 101 markdown docs
- Clone command:
  ```
  git clone --depth 1 https://github.com/reactjs/react.dev.git react-dev-current
  ```

## Verified Counterfactual Pairs (Feature 7 test data)
Confirmed to resolve to real, non-empty documents as of this build:

| Concept | Legacy doc id | Current doc id |
|---|---|---|
| Component state basics | `legacy_state-and-lifecycle` | `current_state-a-components-memory` |
| Context API | `legacy_context` | `current_passing-data-deeply-with-context` |
| Refs to DOM nodes | `legacy_refs-and-the-dom` | `current_useRef` |
| React.Component base class | `legacy_react-component` | `current_Component` (current docs explicitly frame this as legacy-only) |

Note: legacy doc IDs come from the YAML frontmatter `id` field, not the filename —
these can differ (e.g. file `reference-react-component.md` has `id: react-component`).

## Ground-Truth Dataset Specification

The ground-truth dataset (`data/ground_truth/qa_pairs.json`) consists of 100 human-annotated pairs:
- **Coverage**: 50 questions spanning legacy class component patterns (React <= 17) and 50 spanning modern Hooks patterns (React 18+).
- **Schema**: Each record specifies `id`, `question`, `answer`, `gold_doc_id`, `gold_chunk_ids`, `query_type`, and `difficulty_level`.
- **Integrity**: Ground-truth IDs are validated against corpus chunk IDs at build time to prevent evaluation leakage.
