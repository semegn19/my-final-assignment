# Evaluation report

**Filled by:** session 7 (the baseline, and the evaluator's weakness), session 9
(failures named from traces), session 14 (one fix, measured after).

Every number below has the command that produced it, the commit it ran on, and
the model. A number without its command is an impression, and this file holds
none. CI has no keys, so any number CI printed is the offline fake model's.

## Before

- model: qwen/qwen3.8-27b
- commit: 554b03b (`final_assignment/agent.py` modified in the working tree, before the fix)
- command: `uv run bootcamp final grade --agent final_assignment/agent.py --name "Semegn Mulugeta"`
- result: `score: 10/10 (100%) — pass bar 30% — PASSED` — and the gate could not see rank 1: `uv run python final_assignment/debug_paraphrase.py` reported **5 of 7 paraphrases failing** (wrong document or no document), including the course's own breaking query citing mcp-overview at 12.39.

### The evaluator's weakness (session 7)

The pass condition checks that a citation is present and the output shape is
right; it never checks that the answer text is supported by the cited document,
so a canned citation passes. Evidence: the cite-everything fake — the same
answer to every question, citing `rag-basics` at confidence 0.9 — still scored
**50%** (`the cheat scored: 50%`, session 7; `pass_rate` 0.5 in ch07-e3), whose
recorded weakness reads: "The pass condition only verifies that a citation
appears and the output shape, but it does not check that the answer text is
supported by the cited document or that citations match specific claims; this
allows trivial canned citations to pass." The same blind spot in miniature:
this agent scored 10/10 both before and after the rank-1 fix, because no
practice question is a zero-overlap paraphrase — the gate structurally cannot
see that failure.

### Failures, named from traces (session 9)

| Case | Bucket | The trace line that decided it |
|---|---|---|
| grounded case, plain FakeLLM (session 9's own record) | instruction_following | `[decision ] answered with citations []` — retrieval returned rag-basics top-3, the model still answered with no citations |
| "How do I cut up long texts so a model can use them?" (zero-overlap paraphrase) | retrieval_miss | `[retrieve ] top_k=3 -> [('mcp-overview', 0), ('mcp-overview', 2), ('agent-loops', 2)]` — the chunking question's top document was mcp-overview, never rag-basics |

## After

The fix for rank 1 of [ISSUES.md](ISSUES.md) (session 14).

- model: qwen/qwen3.8-27b
- commit: 730521d — the same HEAD as Before; the query-expansion fix is in the working tree
- command: the same command as Before: `uv run bootcamp final grade --agent final_assignment/agent.py --name "Semegn Mulugeta"`
- result: `score: 10/10 (100%) — pass bar 30% — PASSED`, and the same battery now reports **7 of 7 paraphrases reaching their document** (`uv run python final_assignment/debug_paraphrase.py` → `paraphrase checks: ALL PASS`), every refusal guard score unchanged (0.00 / 6.62 / 7.58 / 8.83).
- regression test: `test_regression_rank_1_of_the_issue_list` in `tests/test_contract.py` — `uv run pytest -k regression` → 1 passed (red before the fix: it cited mcp-overview)

### What got better (session 7's `improvement`)

The paraphrase battery went from 2 of 7 to 7 of 7 — the breaking query from mcp-overview at 12.39 to rag-basics at 29.98 — while the practice score stayed 10/10 and every refusal guard's score was unchanged.

### What got worse, or could (session 7's `regression_or_risk`)

The expansion dictionary is hand-kept and will rot as the corpus grows: a new broad key could lift an unrelated question across a floor (session 7's own caveat — "hand-kept lists rot; broad keys poison unrelated queries"), which is why every value's chunk-df and owning documents must be re-audited with `debug_paraphrase.py` whenever a document is added or edited.
