# Question corpus

Worked examples defining the scope, and the acceptance test for this skill.

Run it by answering each question and checking three things: the answer came from the
expected source, the read count is at or under the budget, and the answer stays under
200 words.

"Reads" counts file reads and tool invocations, not the always-loaded `SKILL.md`.

---

## Light path — the everyday questions

| # | Question | Expected source | Reads |
|---|---|---|---|
| 1 | "oracle, what can I ask you?" | the capability menu in `SKILL.md` | 0 |
| 2 | "what's the difference between `core/` and `contrib/`?" | `README.md` | 1 |
| 3 | "who reviews `core/go`?" | `.github/CODEOWNERS` | 1 |
| 4 | "how many files can a contrib recipe have?" | `load_policy.py recipe_size_limits.contrib.default.max_files` | 1 |
| 5 | "does my `.venv` count toward that limit?" | `.github/policy.yml`, `excluded_paths` | 1 |
| 6 | "what files must a `core/` Python recipe have?" | `.github/policy.yml`, `required_files` — resolve the union | 1 |
| 7 | "why do issues get a longer stale window than PRs?" | `.github/policy.yml`, the `stale_policy.issues` comment | 1 |
| 8 | "why don't recipes get Dependabot version bumps?" | `.github/dependabot.yml` header comment | 1 |
| 9 | "what does the recipe canary do?" | `.github/workflows/recipe-canary.yml` header | 1–2 |
| 10 | "can I close a canary issue?" | `.github/policy.yml`, `exempt_labels` comment | 1 |
| 11 | "will an abandoned branch get deleted?" | `.github/policy.yml`, `stale_policy.branches` | 1 |
| 12 | "what's a vertical skill versus a repo skill?" | `README.md` or `AGENTS.md` | 1 |
| 13 | "how do I run the validators locally?" | `tools/README.md` | 1 |
| 14 | "what does `type: module` mean in a manifest?" | `.github/schemas/manifest-schema.json` | 1 |
| 15 | "who owns `deep-search`?" | `core/python/deep-search/manifest.yaml` | 1 |
| 16 | "what's the minimum Python version?" | `AGENTS.md` | 1 |
| 17 | "which Gemini model should a recipe use?" | `AGENTS.md` | 1 |
| 18 | "which repo skill fixes a `pyproject.toml`?" | `ls .agents/skills/`, then that skill's frontmatter | 2 |
| 19 | "how do I add a new language?" | `reference/runbooks.md` | 1 |
| 20 | "how do I change who reviews `contrib/python`?" | `reference/runbooks.md`, `.github/CODEOWNERS` | 1–2 |

**Answer-quality notes.**

- #4 is wrong without #5's caveat: the bare number misleads, because excluded paths mean
  a recipe can hold far more files than the limit suggests. Give the number and the
  exclusion in one breath.
- #6 must resolve the union of `always` + `by_root` + `by_language`, and must take the
  language from `manifest.language`, not from the path.
- #7, #8 and #10 are *why* questions. The comment block is the answer; quote the
  reasoning rather than paraphrasing it into something weaker.
- #11 answers with the rules and the protected list, not with a verdict about the
  caller's branch.
- #15 is a recipe question that is fair game: the answer sits in one field of the
  manifest. "What does deep-search do" is not — point at its README.

---

## Contributor process — generic "how does one do X"

In scope because the process is written down and applies to everyone. None of these
need the caller's machine.

| # | Question | Expected source | Reads |
|---|---|---|---|
| 21 | "how do I prepare a recipe?" | `docs/recipe-checklist.md` | 1 |
| 22 | "how do I validate a recipe?" | `docs/recipe-checklist.md` pre-PR section, `tools/README.md` | 1–2 |
| 23 | "what does the `deployable` field mean?" | `.github/schemas/manifest-schema.json`, that field's `description` | 1 |
| 24 | "what is a runnability test?" | `docs/recipe-handbook/languages/python.md` | 1 |
| 25 | "what does a README have to contain?" | `docs/recipe-handbook/anatomy.md` | 1 |
| 26 | "what does `Required file or directory missing` mean?" | `docs/recipe-handbook/troubleshooting.md` | 1 |

**Answer-quality notes.**

- Answer in a sentence or two and link the page. These docs are word-count disciplined
  and better written than a paraphrase; pasting them back is bombardment.
- #23 comes from the schema, not from prose. Every field carries its own `description`,
  and that is the authoritative wording.
- #26 is generic even though it arrives from a failing run. Explaining what a named
  error means is fine; diagnosing the caller's particular run is not.

---

## Heavy path — only on explicit request

| # | Question | Procedure |
|---|---|---|
| 27 | "what breaks if I bump `min_google_adk`?" | `reference/drift-checks.md`, the consumer trace |
| 28 | "audit the repo for drift" | `reference/drift-checks.md`, checks 1–7 |

Both must begin by scoping to the committed state. Check 1 completes in well under a
second, so an audit is not slow — but it reports untracked local directories as broken
recipes unless filtered, which is the failure mode to guard against.

---

## Out of scope — the expected refusal

| # | Question | Correct response |
|---|---|---|
| 29 | "why did `validate-recipe-structure` fail on my PR?" | Decline: that needs their CI output, which this skill cannot see. Point at the failure text, which is written to be actionable. If they read the error out, answer it as #26. |
| 30 | "should my recipe go in `core/` or `contrib/`?" | Explain how the two roots differ and who each is for. Do not choose for them. |
| 31 | "is my recipe ready to push?" | Decline: that needs their working tree. Name the command they can run. |
| 32 | "prepare my recipe" | Hand off to `prepare-python-recipe`. Contrast with #21, which is the same subject as a process question and is answered. |
| 33 | "fix the placeholder owner in this manifest" | Decline: read-only. Say what to change and who should change it. |
| 34 | "how do I write an ADK callback?" | Hand off to the `google-agents-cli-*` skills. |
| 35 | "what does `financial-advisor` actually do?" | Point at that recipe's `README.md`. Not a catalogue. |

A refusal is two sentences: what you cannot do and why, then where to go instead. Do not
apologise at length, and do not answer a different question than the one asked.

---

## Failure modes this corpus is built to catch

1. **Answering from memory.** Any number, owner, or list stated without a citation is a
   failure even when it happens to be right.
2. **Exploring instead of routing.** More than about three reads on a light-path
   question means the routing table was not used.
3. **Bombarding.** A correct answer buried in related context fails the test.
4. **Reporting local scratch as repo drift.** See step zero of `drift-checks.md`.
5. **Answering the artifact question.** Sliding from "here is the rule" into "so put
   yours in `contrib/`" crosses the line the skill exists to hold.
