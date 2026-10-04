# my-final-assignment

A research assistant that answers developer questions **only** from the six documents in `data/corpus/`, names the document it used, and refuses — flagged for human review, citing nothing — whenever those sources don't support the question.

![check](https://github.com/semegn19/my-final-assignment/actions/workflows/check.yml/badge.svg)

## The problem

Developers debugging with internal docs ask questions in their own words, but grep needs the document's words, and a bare LLM invents confident answers from memory with no way to tell which is which. A paraphrase gets an unearned refusal; a plausible-sounding guess gets an unearned yes. Minute 1 of the demo is exactly that fork: one question gets a grounded, cited answer, one gets an honest "I don't know" — and the trace proves which path was taken.


## Demo

Two runs, pasted exactly as the commands printed them. Never an edited one.
`trace` prints every step the agent took, then the answer.

### One supported answer

```bash
uv run bootcamp capstone trace "How does chunking work in RAG?"
```

```text
[retrieve] top_k=6 -> [('rag-basics', 1), ('rag-basics', 2), ('evaluation-basics', 0), ('prompt-injection', 2), ('prompt-injection', 1), ('rag-basics', 0)]
[llm_call] attempt 1: 121 chars
[decision] refusal-shaped reply at 11.94 >= 10.0: quoting rag-basics verbatim

answer: Grounded in the corpus — quoting rag-basics (RAG Basics) verbatim:

# RAG Basics

Retrieval-augmented generation (RAG) grounds a model's answer in documents you
control. Instead of hoping the model memorized the right facts, the application
retrieves relevant passages at question time and places them in the prompt. The
model's job shrinks from "know everything" to "synthesize from what you were
shown" — a job it is much better at.

## The pipeline

A minimal RAG pipeline has four stages. **Chunking** splits documents into
passages small enough to be individually relevant — respecting paragraph
boundaries beats cutting at a fixed character count mid-sentence. **Indexing**
prepares chunks for search; a lexical index (keyword overlap) is a perfectly
respectable baseline and is deterministic, cheap, and debuggable. Embedding-based
semantic search improves recall on paraphrased questions but adds a dependency
and a failure mode you cannot inspect by eye. **Retrieval** selects the top-k
chunks for a query. **Generation** answers using only the retrieved context, with
citations naming the source documents.

## Retrieval failure vs generation failure

When a RAG system answers wrongly, the first diagnostic question is: did the
right passage reach the prompt? If retrieval missed it, no amount of prompt
engineering will fix the answer — fix chunking, the index, or the query instead.
If the right passage was present and the answer still went wrong, the failure is
in generation: instructions, formatting, or the model ignoring context. Keeping
retrieved chunks inspectable (log them, show them) is what makes this diagnosis a
two-minute check instead of a guessing game.

## Citations keep the system honest

Every answer should name the document ids it drew from, and the application
should verify those ids against what was actually retrieved. A citation the
retriever never returned is a fabrication and should be stripped and flagged.
The "I don't know" case matters most: when nothing relevant is retrieved, the
system must refuse rather than let the model improvise from its training data.

## Small corpus, high leverage

Most of RAG's value appears with a corpus of five documents, not five million.
Start small, keep the corpus versioned in git, and grow it only when a measured
retrieval failure says you need to.
citations: ['rag-basics']
confidence: 0.5
needs_human_review: False
```

### One refusal

```bash
uv run bootcamp capstone trace "What is the capital city of Mongolia?"
```

```text
[decision] no relevant chunks; refusing without an LLM call

answer: I don't know based on the provided corpus.
citations: []
confidence: 0.0
needs_human_review: True
```


## Architecture

One question costs **0 or 1 model calls — at most 2**. It's a chain with a
refusal branch: retrieval scores every chunk against the question first, and a
best score under the gray floor (3.0) refuses *before* any call is spent.
Otherwise one model call runs (plus at most one corrective retry if the reply
isn't valid JSON), its reply is treated as untrusted data — citations verified
against what retrieval actually returned (fabrication → strip and flag),
reply-shaped directives in retrieved text flagged, never obeyed — and a
four-band decision (3.0 / 6.0 / 10.0 floors) answers, quotes the source, or
refuses. Provider error or timeout is a flagged refusal, never a raised
exception. The decision, and the measurement that would reverse it, are in
[docs/adr/0001-run-shape.md](docs/adr/0001-run-shape.md).

See [docs/adr/0001-run-shape.md](docs/adr/0001-run-shape.md).

## Measured results

Every number here comes from a command in this table, run on this commit. Say
which model produced it: CI has no keys, so a CI number is always the offline
fake model's.

| What | Command | Model | Result |
|---|---|---|---|
| Contract tests | `uv run pytest` | qwen/qwen3.8-27b | 4 passed, 2 skipped, 3 xpassed in 0.46s |
| Practice grader | `uv run bootcamp capstone grade` | qwen/qwen3.8-27b | score: 10/10 (100%) — pass bar 30% — PASSED |
| Evaluation, before and after | see [docs/EVAL_REPORT.md](docs/EVAL_REPORT.md) | qwen/qwen3.8-27b | paraphrase battery `2/7 → 7/7` passing, practice score unchanged at 10/10 |

## The honest limitation

Rank 1 of the issue list — lexical retrieval misses paraphrases with zero word
overlap, so a supported question in fresh words earns a refusal — was fixed
narrowly by a hand-kept query-expansion map (`_EXPANSION`, 18 keys) that will
rot as the corpus grows; next step is re-auditing every value's chunk-df with
`debug_paraphrase.py` on each corpus change, and eventually replacing hand-kept
keys with a retrieval method that generalizes.

The full ranked list is in [docs/ISSUES.md](docs/ISSUES.md).

## How to run it

```bash
git clone https://github.com/semegn19/my-final-assignment && cd my-final-assignment && uv sync && uv run pytest
```

No key needed: without a `.env` it runs on the offline fake model. For a real
model, copy `.env.example` to `.env`, fill in your provider, and
`uv sync --extra anthropic` (or `--extra openai`).

To hand in the final assignment, commit and push, then run
`uv run bootcamp capstone submit --github <you>`. It runs the practice set
first, then answers the final questions and opens the pull request.
`--dry-run` shows the bundle without handing anything in.

---

| Path | What it is |
|---|---|
| `agent.py` | The agent: `YourAgent`, the class the tests, `trace` and the grader run |
| `tests/test_contract.py` | The capstone contract, as tests (`uv run pytest -k refusal`, `-k injection`, ...) |
| `data/corpus/` | The six source documents, versioned; nothing here writes to them |
| `docs/EVAL_REPORT.md` | Numbers you produced, before and after, with the command behind each |
| `docs/SKILL.md` | A skill another assistant can load (session 10) |
| `docs/adr/0001-run-shape.md` | The architecture decision and what would reverse it (session 10) |
| `docs/RETENTION.md` | What a session remembers, and what it refuses to (session 11) |
| `docs/ISSUES.md` | The ranked issue list (session 9, kept until 14) |

Built during the Dev3Pack AI Engineering bootcamp, on the course package at
commit `1de8649529156c35bdeafacfbbef40726a9b0fba` of https://github.com/Gecko-Academy/dev3pack-cohort-2026-09.
