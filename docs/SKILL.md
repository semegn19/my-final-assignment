---
name: pre-pr-diff-review
description: Use when reviewing a diff in src/bootcamp_agent before opening a PR — severity-ranked findings with the smallest failing example; NOT for writing new features.
---

# Skill

**Filled by:** session 10. The five sections are the ones `ch10-e1` reads, and
the evidence below is the before-and-after pair of runs you saved.

## When to use (`when_to_use`)

Reviewing a diff in `src/bootcamp_agent` before I open a PR. NOT for writing
new features.

## Workflow (`workflow`)

1. Read the diff only.
2. Identify behavioral changes and the smallest failing example.
3. Classify findings by severity: critical, major, minor.
4. Suggest a minimal reproducible test or a one-line fix.
5. Re-check the diff after the suggested change.

## Output format (`output_format`)

```text
- Severity: <critical|major|minor>
- Location: file:line
- Summary: one sentence
- Suggestion: concise code change or unit test to add
```

## Failure rules (`failure_rules`)

Critical: removes or disables tests, introduces silent data-loss, or adds
secrets. Major: behavior changes without tests or ambiguous API changes.
Minor: style, docs, or formatting only. Always escalate when uncertain.

## Safety boundary (`safety_boundary`)

Do not run code, do not propose or include secrets, do not edit the repo —
only suggest changes and tests. Avoid behavioral fixes that require elevated
permissions.

## Evidence

### Without the skill (`without_skill`)

```text
Assistant: 'Looks fine' — missed that the diff removed an input validation
and a unit test, which would let malformed input crash production.
```

### With the skill (`with_skill`)

```text
Assistant findings:
- major: src/bootcamp_agent/parser.py:42 -> input validation removed; add unit test asserting invalid input raises ValueError.
- minor: README.md formatting; harmless.
Suggestion: add one failing unit test and a 2-line validation guard in parser.py.
```

### The instruction you fixed (`improved_instruction`)

When behavior changes, require at least one targeted unit test demonstrating
the failing case (name + one assertion) — the first version accepted a fix
with no test, and the same diff regressed again on the next review.
