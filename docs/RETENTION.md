# Retention policy

**Filled by:** session 11. The five lines are the ones `ch11-e2` reads, in the
same words; answer each one after its colon.

STORED: one answer_style preference; the last 5 question summaries (episodes).

WHY: it keeps responses short when asked and preserves recent context for this session.

CORRECTED BY: the learner updating preferences or calling reset(); stale episodes are trimmed automatically.

EXPIRES: after 5 new episodes, or immediately on reset().

WE REFUSE TO REMEMBER: personal data, secrets, credentials, tokens, and anything without a clear owner or expiry.

## How the code enforces it

The designated test is `test_memory_is_capped_reset_and_kept_per_user` in
`tests/test_contract.py`, which must prove the cap holds at 5 episodes after 7
questions, that `reset()` leaves neither preferences nor episodes behind, and
that one user's memory never answers another's. Today it is session 11's
`skip` placeholder — `YourAgent` holds no session state yet, so the same cap
and reset are currently verified by the `ch11-e1` probe in session 11's
notebook (7 questions in → exactly 5 episodes; `reset()` → both collections
empty). The test body gets written the moment this agent gains memory.
