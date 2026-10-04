# ADR 0001: the shape of one run

**Filled by:** session 10, for the choice you measured in session 8 (chain,
loop or graph, and the model calls each one cost). The four fields are the
ones `ch10-e2` reads.

- Status: accepted
- Date: 2026-10-04

## Context

The capstone task has one fixed shape — retrieve, answer once, name the
document, refuse what the corpus does not support — and session 8 measured
the three candidate shapes against exactly that task before choosing. The
chain costs **1 model call** and cannot refuse before it spends it, so bad
retrieval flows straight into the answer. The tool loop costs **1–2 calls
(+retry)** and can refuse before the call. Reflection costs **2–3 calls** for
the critique plus the revision. The graph-shaped four-role script
(draft/critique/revise) costs **3 model calls** and needs a runtime dependency
the course does not ship (`uv add langgraph  # not a course dependency`).
Two numbers forced the choice: the contract requires an unsupported question
to cost **0 model calls**, and the measured loop covers every branch this run
actually has — refuse, answer in one pass, correct one malformed reply.

## Decision (`decision`)

We keep the hand-written loop in `agent.py`: refuse before the model call
when retrieval supports nothing, and cap every question at two model calls —
one call plus one corrective retry — with no graph framework.

## Options considered (`options_considered`)

1. The hand-written loop in `agent.py`: retrieval scores first, a refusal
   path that spends zero calls, then one bounded call with one corrective
   retry.
2. A graph framework (LangGraph-style state machine) wired to the same
   `LLMClient` seam, with declared nodes for retrieve, call, parse, verify.

## Why not the other option (`why_not`)

The graph's own measured script spends 3 model calls where the loop spends
1–2, so every question would pay for roles (draft, critique, revise) the
six-document corpus never asks for, plus a runtime dependency and a second
vocabulary of nodes and edges to learn, test and debug — for a run whose only
real branch is the refusal, which the loop already takes before any call is
spent.

## What would reverse it (`reverses_it`)

When a question needs more than 2 model calls in 10 of the golden cases —
today a refusal costs 0 calls, the normal answer 1, and the corrective retry
caps the worst case at 2, so 10 cases showing 3 calls each would mean the
single pass no longer fits the questions.
