# Ranked issues

**Filled by:** session 9 (the first list, `cap01-e5`), kept current until
session 14, which fixes rank 1 and adds its regression test.

At least three rows. Ranks 1, 2, 3... with no gap and no tie: two issues ranked
1 is a list nobody prioritised. The impact is what orders it.

The columns are the three fields `cap01-e5` reads.

| rank | issue | impact |
|---:|---|---|
| 1 | Lexical retrieval misses paraphrases with no word overlap | a user who asks in their own words gets a refusal for a supported question |
| 2 | The evaluator checks which doc was cited, not whether the answer is faithful | a weak answer can still look good when the citation is correct but the claim is wrong |
| 3 | Nothing bounds a provider that hangs rather than failing | a stalled provider can block the user with a timeout instead of an honest error or fallback |

## Rank 1, in progress

- The fix: query expansion at the retrieval boundary — `agent.py`'s `_EXPANSION`
  maps a user's word to the corpus's word (narrow keys; every value audited for
  chunk-df and owning documents), applied by `_expanded_query()` inside
  `_document_scores()` and `_context()` so scoring and context see the same
  query.
- The regression test: `test_regression_rank_1_of_the_issue_list` in
  `tests/test_contract.py` — red before the fix (it cited mcp-overview), green
  after (it cites rag-basics).
- Before and after: see [EVAL_REPORT.md](EVAL_REPORT.md).
