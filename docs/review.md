# Docs Review

> **Scope:** `docs/README.md`, `docs/recipe-checklist.md`,
> `docs/recipe-handbook/README.md`, `docs/recipe-handbook/anatomy.md`,
> `docs/recipe-handbook/skills-catalog.md`,
> `docs/recipe-handbook/troubleshooting.md`,
> `docs/recipe-handbook/languages/python.md`.
>
> **Structure:** seven personas, each in first-person voice. Every issue has a
> file reference and a concrete fix description so an acting agent can apply
> it without ambiguity.

---

## Persona 1 — First-Time Contributor

*I'm a developer who wants to contribute a recipe for the first time. I know
what ADK is. I'm starting at `docs/README.md` and working my way through.*

I land on `docs/README.md`. It's 75 words. It tells me to go to the
checklist. But I don't yet know what a "recipe" is. The word appears three
times on that page and is never defined. I find the glossary eventually — at
the bottom of `docs/recipe-handbook/README.md` — but I've already been
reading for several minutes using a term I don't fully understand.

The checklist opens with an "AI skills reference" table before telling me
anything about the contribution process. I don't know if I need an AI
assistant, whether the table applies to me, or whether I can skip it. There's
a one-liner at the bottom of that section saying "if you're not using an AI
assistant, § 3 below shows the manual sequence" — but I almost missed it.

The handbook README tells me to start with anatomy, then python.md, then a
linked walk-through called "New Python recipe from scratch." I click that
link. Nothing happens — the anchor does not exist in `python.md`. The actual
section is called "The fast path." I'm left wondering if I'm missing a page.

I fill in `manifest.yaml` and I see a field called `poc`. I have no idea what
this abbreviation means. I guess "Point of Contact" from context, but it's
never written out anywhere.

I try to understand the `deployable` field in the manifest table and see it
mentions "Agent Garden." There's no explanation of what Agent Garden is and
no link.

### Issues

**P1-1** · `docs/README.md`
`docs/README.md` contains no definition of "recipe." The term is used
immediately and repeatedly. **Fix:** add one sentence before the "Start here"
section, e.g., "A recipe is a runnable agent example (or importable agent
module) that lives under `contrib/`."

**P1-2** · `docs/recipe-checklist.md` — AI skills preamble
The "AI skills reference" section does not explain what an AI assistant is,
whether one is required, or what to do without one. The fallback note ("§ 3
below") is a single line at the bottom of the section and easy to miss.
**Fix:** add a lead sentence at the top of the section, e.g., "These skills
run inside an AI coding assistant (e.g. CloudCode). If you don't have one,
skip to [§ 3](#3-before-you-open-the-pr) for the manual commands."

**P1-3** · `docs/recipe-handbook/README.md`, line 37
Broken anchor link: `./languages/python.md#new-python-recipe-from-scratch`
resolves to nothing. The target section in `python.md` is `## The fast path`,
whose anchor is `#the-fast-path`. **Fix:** change the link target to
`./languages/python.md#the-fast-path` and update the link text from "New
Python recipe from scratch" to "The fast path — new and existing recipes."

**P1-4** · `docs/recipe-handbook/anatomy.md`, line 55
`ownership.poc` is never expanded. **Fix:** change the field label in the
table to `ownership.poc` (Point of Contact) so readers don't have to guess
the abbreviation.

**P1-5** · `docs/recipe-handbook/anatomy.md`, line 61
`deployable` description references "Agent Garden" with no explanation or
link. **Fix:** add a brief inline explanation, e.g., "Agent Garden is the
Google discovery surface for deployed ADK agents" or add a link to relevant
documentation. If the term is internal-only and not publicly documented,
remove the reference from the description.

**P1-6** · `docs/README.md` vs `docs/recipe-handbook/README.md`
Conflicting entry-point instructions. `docs/README.md` says "Go straight to
the checklist." The handbook README says "If you're new here, start with:
anatomy → python → fast path." A first-timer doesn't know which to follow.
**Fix:** pick one primary entry point. Recommended: keep the checklist as the
entry point and reframe the handbook README's "start with" list as "after you
try the checklist, read these for deeper context."

**P1-7** · `docs/recipe-checklist.md`, lines 62–63
"Java / Go / TypeScript / Kotlin — language-specific guidance coming soon."
A first-timer using those languages is left with no actionable path. **Fix:**
replace the placeholder with specific guidance: state which structural checks
already apply (they do — via `uv run validate`) and link to the anatomy page
for the rules that are language-agnostic.

---

## Persona 2 — Returning Developer

*I contributed a recipe several months ago. Now I need to update it to meet
current repo standards. I'm looking for what changed and what I need to fix.*

My first instinct is to scan the checklist for anything new. There is no
changelog, no "updated" date, no "what's new" section on any page. I have to
read the entire checklist and mentally diff it against what I remember. I'm
going to miss things.

I almost miss the model deprecation warning. It's buried as a bullet under
"Python" in section 2: "use up-to-date models like `gemini-3.5-flash` (not
`gemini-2.0-flash` or `gemini-2.5-flash`)." I submitted my last recipe using
`gemini-2.0-flash`. If I hadn't been reading carefully, I'd have opened a PR
and waited for CI to tell me.

I want to run `prepare-python-recipe` on my existing recipe but I'm nervous.
Will it overwrite my code? I find the reassurance ("safe to re-run and won't
overwrite your `.env.example` or hand-written Python code") — but only in
`python.md`, at the very bottom of the page. It's not in the skills catalog
entry for `prepare-python-recipe`, which is where I actually looked first.

The troubleshooting page is comprehensive but undated. I can't tell which of
these errors are newly introduced CI checks and which have been there for
years.

The handbook README opens with "You're here to contribute a recipe. Welcome."
That's for new contributors. There's no path for me — someone returning to
bring an old recipe up to standard.

### Issues

**P2-1** · All docs — Missing changelog
No "what changed" section exists anywhere. **Fix:** add a `## What changed
recently` section to `docs/recipe-checklist.md` at the top (above section 1)
listing recently added requirements with brief descriptions. Alternatively,
create `docs/CHANGELOG.md` and link to it from the checklist.

**P2-2** · `docs/recipe-checklist.md`, section 2 (Python)
The model deprecation notice is a checklist item formatted identically to all
other items. It's easy to skip. **Fix:** pull the deprecation notice out of
the bullet list and make it a visible callout directly above the Python
checklist items, e.g.:
> **Deprecated models:** `gemini-2.0-flash` and `gemini-2.5-flash` are no
> longer accepted. Use `gemini-3.5-flash`.

**P2-3** · `docs/recipe-handbook/skills-catalog.md` — `prepare-python-recipe` entry
The entry says nothing about safety for existing recipes. **Fix:** add to the
entry's bullet list: "Safe to re-run on existing recipes — won't overwrite
`.env.example` or hand-written Python code."

**P2-4** · `docs/recipe-handbook/README.md`, introduction
The opening paragraph addresses only new contributors. **Fix:** add a second
path explicitly for returning contributors, e.g.: "Updating an existing
recipe? Run `prepare-python-recipe` against your recipe path — it's safe to
re-run and will apply any new requirements automatically. Then check the
checklist for any manual steps."

**P2-5** · `docs/recipe-handbook/troubleshooting.md` — No recency signals
Readers can't distinguish new CI checks from established ones. **Fix:** where
a CI check was recently added, annotate the section header with "(added
[month year])" so returning contributors know at a glance what's new.

---

## Persona 3 — "Just Get It Done" Developer

*I want to prepare my recipe for the repo as fast as possible. I'm not going
to read everything. Point me at the commands.*

I open `docs/README.md`. It tells me to go to the checklist. Fine. I open the
checklist. The first thing I see is a table of AI skills. I scan it. There's
one skill that does everything: `prepare-python-recipe`. Great, I run it. But
wait — do I even have the AI assistant this requires? That's not clear. I
scroll past the table.

Now I'm in section 1, section 2, section 3. I'm reading about structural
checks, lint, tests, integration tests. I just want the one-liner block. I
keep scrolling. I finally find the "Full manual command sequence" — an
all-in-one paste block. It's at the bottom of section 3. Why is it at the
bottom?

I find the `python.md` fast path section, which is exactly what I needed —
two prompts and I'm done. But it's also at the bottom of the page. Everything
useful is hidden at the bottom of things.

The "Set your recipe path" step uses a 4-space indented code block. The
"Full manual command sequence" uses a fenced code block with bash syntax
highlighting. These look different in my editor. Pick one style.

### Issues

**P3-1** · `docs/recipe-checklist.md` — "Full manual command sequence" placement
The all-in-one paste block is the single most useful piece of content on the
page for this persona. It is currently the last item in section 3. **Fix:**
promote it. Add a "Quick start" or "TL;DR" section at the very top of the
checklist (above the AI skills table) containing just the path-setting export
and the command block. Keep the detailed sections below for reference.

**P3-2** · `docs/recipe-handbook/languages/python.md` — "The fast path" placement
"The fast path" section is the most actionable content in `python.md` but
appears last. **Fix:** move it immediately after the "Package layout" section,
before "Copy-paste starters." Readers who want to act fast should hit it
before they get to file templates.

**P3-3** · `docs/recipe-checklist.md` — Mixed code block style
The "Set your recipe path" subsection uses a 4-space indented block; every
other command example uses fenced ``` blocks. **Fix:** convert the 4-space
indented block at line 75 to a fenced bash block:
```
```bash
export RECIPE_PATH=contrib/python/my-recipe
```
```

**P3-4** · `docs/recipe-checklist.md` — AI skills table has no "requires AI assistant" signal
The table implies `prepare-python-recipe` is always available, but it requires
a specific AI assistant. A reader without one will be confused when the
trigger phrase does nothing. **Fix:** add a one-line note above the table:
"Requires an AI coding assistant (e.g. CloudCode). Skip to § 3 for manual
commands."

**P3-5** · `docs/recipe-handbook/skills-catalog.md` — "How to invoke a skill" code blocks
The three example prompts use 4-space indented blocks, not fenced code blocks.
**Fix:** convert to fenced blocks for visual consistency with the rest of the
docs.

---

## Persona 4 — Non-Native English Speaker

*English is my second language. I can read technical documentation, but
idiomatic expressions slow me down and unclear abbreviations stop me cold.*

Overall the writing is clear. Short sentences help. But there are several
points where I had to pause.

"Recipes must earn their place" — I understand "earn their keep" and "earn a
place," but this exact phrasing is slightly unusual. I understand the meaning
from context but it took me a moment.

"Ship a recipe" — Used in the checklist and the skills catalog. In my
language, "ship" means to send a physical package. I know it's tech slang for
"release" but it's informal and not universal.

"CI red on your PR" — "Red" meaning "failing." The color metaphor is not
obvious if you haven't used GitHub CI before. Someone new to GitHub Actions
may not make this connection.

"description proxy" in `anatomy.md` — The word count is described as a
"description proxy." I understand "proxy" in technical contexts (network
proxy, proxy variable) but using it this way — meaning the word count stands
in as a signal of description quality — is unusual phrasing.

`poc` — Never expanded anywhere. I guessed from context.

Sentence fragments in troubleshooting: "Missing entirely: run
`generate-manifest`." Complete sentences would help.

### Issues

**P4-1** · `docs/recipe-handbook/README.md`, line 15
"Recipes must earn their place." → **Fix:** rewrite as "Every recipe must
have a clear purpose and offer something new to the ADK community."

**P4-2** · Multiple docs — "ship a recipe"
Used in `docs/recipe-checklist.md` (line 5) and `docs/recipe-handbook/skills-catalog.md`
(line 9). **Fix:** replace "ship" with "submit" or "publish" throughout.

**P4-3** · `docs/recipe-checklist.md`, section 4 — "CI red on your PR"
"CI red on your PR?" is color-coded jargon. **Fix:** change to "CI failing on
your PR?" which is unambiguous regardless of familiarity with GitHub's green/
red status indicators.

**P4-4** · `docs/recipe-handbook/anatomy.md`, line 98
"(description proxy)" — the parenthetical is unclear. **Fix:** replace with
"(the word count is used as a minimum quality signal)."

**P4-5** · `docs/recipe-handbook/anatomy.md`, line 55
`poc` is never expanded. **Fix:** write `poc` (Point of Contact) the first
time it appears. (Mirrors fix P1-4 — same location, same fix.)

**P4-6** · `docs/recipe-handbook/troubleshooting.md`, line 12
"Missing entirely: run `generate-manifest`." is a sentence fragment. **Fix:**
rewrite as a complete sentence: "If `manifest.yaml` is missing entirely,
run `generate-manifest`." Apply the same fix to all other fragment-style
entries in troubleshooting.md (lines 12, 28, 37, 44, 51, 58, 65, 107, etc.).

**P4-7** · `docs/recipe-handbook/README.md`, line 11
"return when a checklist item needs unpacking" — "unpacking" is idiomatic.
**Fix:** change to "return when a checklist item needs more explanation."

**P4-8** · `docs/recipe-handbook/troubleshooting.md`, lines 168–169
"bypass registry trust" is opaque. **Fix:** rewrite as "skip the package
registry's security verification."

---

## Persona 5 — Tech Writer

*I analyze technical documentation for structure, layout, tone consistency,
and fluidity. I'm not here to check technical accuracy — I'm here to tell you
whether a human can actually read this.*

There's real editorial thought behind these docs. The checklist/handbook split
is a sound information architecture. The navigation footer on every page is
the right habit. The "copy-paste starters" section in `python.md` is
genuinely good — it anticipates what the reader needs and delivers it.

But there are enough inconsistencies to undermine the experience of reading
across pages.

**Tone is not unified.** `skills-catalog.md` is conversational and warm: "Say
the skill's name or its trigger phrase to your assistant." `anatomy.md` is
clinical: table-heavy, terse. `troubleshooting.md` reads like notes written
for the author, not the reader: "Missing entirely: run `generate-manifest`."
`python.md` is friendly and example-driven. These do not feel like the same
document set.

**Code block style is inconsistent.** Some sections use 4-space indented
blocks; others use fenced ``` blocks with syntax hints. Fenced blocks render
better on GitHub (they get a copy button, syntax highlighting) and should be
used universally.

**The navigation footer on `docs/README.md` is misleading.** The footer reads
`← [Checklist] · [Handbook]` — but the `←` implies those are parent pages.
`docs/README.md` is the parent. The back-arrow doesn't belong there.

**`troubleshooting.md` has no table of contents.** It's 238 lines of flat
`##` headers. A reader arriving mid-crisis (CI just failed) has to scroll the
entire page to find their error. A ToC at the top would solve this in one
edit.

**The `>` blockquote is used for exactly one callout** — the "Fastest path"
note in `skills-catalog.md`. Nowhere else. Either use it consistently for all
tips/callouts, or remove it and use bold text like the rest of the docs.

**HTML comment metadata bleeds into the reading experience.** Every file
starts with `<!-- word count: N (target T, cap C) -->`. This is invisible
when rendered but visible when editing, and it normalizes internal tooling
markers in user-facing files. If these are needed for editorial tooling, move
them to a companion file or use a non-standard metadata format that won't
confuse future editors.

**The checklist section numbering has a gap.** Sections are numbered 1, 2, 3,
4 — but the "AI skills reference" section sits before section 1 with no
number. The flow reads: [unnumbered] → 1 → 2 → 3 → 4. This is jarring.

**No "last updated" signal on any page.** For `troubleshooting.md` especially
— a page readers consult when something breaks — there's no indication of
whether the content reflects the current CI configuration.

### Issues

**P5-1** · All docs — Mixed code block style
4-space indented blocks and fenced ``` blocks are used interchangeably.
Affected locations: `recipe-checklist.md` lines 75, 92–94; `skills-catalog.md`
lines 20–23; `troubleshooting.md` lines 152–153, 198–200. **Fix:** convert
all 4-space indented command blocks to fenced bash blocks. Use plain fenced
blocks (no language hint) for dialogue examples (AI assistant prompts).

**P5-2** · `docs/README.md`, line 21
Navigation footer uses `←` on the top-level README, implying the linked
pages are parents. They are not. **Fix:** remove the `←` from the footer on
`docs/README.md` only. Or replace the footer entirely with a simple
"Navigation: [Checklist] · [Handbook]" line without a directional arrow.

**P5-3** · `docs/recipe-handbook/troubleshooting.md` — No table of contents
238 lines, flat `##` header structure, no ToC. **Fix:** add a linked table
of contents at the top of the file, listing every error heading so readers
can jump directly to their failure. This is the single highest-impact change
for `troubleshooting.md`.

**P5-4** · `docs/recipe-handbook/skills-catalog.md`, line 12
`>` blockquote used for "Fastest path" callout but nowhere else in the docs.
**Fix:** either (a) replace the blockquote with bold text to match the rest
of the docs, or (b) adopt the `>` blockquote as the standard callout format
and apply it consistently across all pages for tips, warnings, and "fastest
path" notes.

**P5-5** · `docs/recipe-checklist.md` — Section numbering gap
"AI skills reference" has no number. It sits before section 1 and breaks the
numbered sequence. **Fix:** either (a) renumber it as `## 0. AI skills
reference` to make the gap intentional, or (b) convert it to a callout block
(using `>` or a bold preamble) that clearly sits outside the numbered
checklist flow.

**P5-6** · All docs — Tone inconsistency across pages
`skills-catalog.md` and `python.md` are conversational; `anatomy.md` and
`troubleshooting.md` are terse and clinical. **Fix:** establish a consistent
register. Recommended: short, direct sentences with complete subject-verb-object
structure throughout. The conversational warmth in `python.md` is the target
register — apply it to `anatomy.md` and `troubleshooting.md`.

**P5-7** · All docs — No "last updated" signal
**Fix:** add a `_Last updated: [date]_` line at the bottom of each file,
above the navigation footer. Priority: `troubleshooting.md` first.

**P5-8** · `docs/recipe-handbook/troubleshooting.md` — "Advisory notices" section
The "Advisory notices" section at the bottom is different in kind from the
rest of the page (these don't fail CI) but is formatted identically to error
sections. **Fix:** visually distinguish it — use a `>` blockquote, indent it,
or give it a different header level — so readers understand these will not
block their PR.

**P5-9** · `docs/recipe-checklist.md` / `docs/recipe-handbook/languages/python.md`
Integration test exclusion patterns documented in two places (checklist
section 3 and `python.md` integration tests section) with slightly different
framing. **Fix:** designate one location as the source of truth (recommend
`python.md`) and replace the checklist section with a single cross-reference
link.

---

## Persona 6 — Manager / Director

*I care about quality, consistency, and how this documentation reflects on the
project. I'm evaluating whether these docs are ready to represent the repo.*

The architecture is sound. The checklist/handbook split shows clear intent.
The troubleshooting guide is thorough. The AI skills catalog is a genuinely
good idea for reducing friction. These are real strengths.

But there are gaps that concern me.

**The landing page undersells the project.** `docs/README.md` is 75 words.
By the repo's own CI standard, a recipe README must have at least 100 words.
The docs landing page doesn't even meet the bar it sets for contributors. A
new contributor arriving here for the first time gets no context: what is this
repo, who is the audience for a recipe, why should I contribute?

**The quality bar is hidden.** "Recipes must earn their place." "Have a clear
intent." "Something new to teach." These editorial standards — the ones that
actually differentiate a good recipe from a box-checking exercise — are in the
handbook README, a page most contributors will skim. They're not in the
checklist. A contributor who reads only the checklist will think the bar is
purely mechanical: pass CI, open a PR.

**An internal product URL appears in public-facing docs.** The reference to
"Antigravity CLI" with the link `https://antigravity.google/product/antigravity-cli`
appears in both `recipe-checklist.md` (line 13) and `skills-catalog.md`
(line 6). If `github.com/google/adk-samples` is a public repository, this
Google-internal URL should not appear in its documentation.

**There is no defined support process.** A contributor whose PR is stuck, who
has a question the docs don't answer, or who hits a false positive in CI has
one option: "open a GitHub issue." There's no mention of who reviews `contrib/`
PRs, no expected turnaround, no escalation path. For a project trying to
attract external contributors, this communicates low confidence.

**There is maintenance-risk duplication.** Integration test exclusion patterns
appear in both `recipe-checklist.md` and `python.md`. The `core/` size limits
are omitted from the `anatomy.md` table (only `contrib/` is listed). These
are small gaps now, but they grow over time.

### Issues

**P6-1** · `docs/README.md` — Landing page is too thin
At 75 words, it doesn't meet the 100-word minimum the repo enforces on
contributors. It also provides no context for new arrivals. **Fix:** expand
to at least 150 words. Add: (1) a one-sentence description of the repo's
purpose, (2) a one-sentence definition of a recipe, (3) who the intended
contributors are, and (4) a note on what distinguishes a good recipe from a
minimal one.

**P6-2** · `docs/recipe-handbook/README.md`, "What makes a good recipe"
The editorial quality bar lives in the handbook but not the checklist. **Fix:**
add a brief "Editorial bar" item to `recipe-checklist.md` section 1 ("Always"),
e.g.: "[ ] Recipe has a clear, unique purpose — see [What makes a good
recipe](./recipe-handbook/README.md#what-makes-a-good-recipe)." This makes
the editorial bar unavoidable.

**P6-3** · `docs/recipe-checklist.md`, line 13 and `docs/recipe-handbook/skills-catalog.md`, line 6
"Antigravity CLI" link (`https://antigravity.google/product/antigravity-cli`)
is a Google-internal product URL. **Fix:** replace "Antigravity CLI" and the
internal link with "CloudCode" (or "your AI coding assistant") and either
link to a public product page or remove the link entirely.

**P6-4** · `docs/recipe-handbook/README.md`, "Contact" section
"Open a GitHub issue" is the only support path. There's no mention of PR
review ownership, expected turnaround, or who watches `contrib/`. **Fix:**
add 2–3 sentences clarifying who reviews `contrib/` PRs and the expected
response timeline. If this is not yet defined, state that explicitly so
contributors have accurate expectations.

**P6-5** · `docs/recipe-handbook/anatomy.md`, size limits table
The table lists limits only for `contrib/`. There is no row for `core/`,
leaving readers to wonder if `core/` has different limits or none at all.
**Fix:** add a row for `core/` with its limits, or add a note below the table
stating that `core/` recipes are managed by the `agents-cli` team and are not
subject to the same contributor-facing limits.

**P6-6** · `docs/recipe-checklist.md` / `docs/recipe-handbook/languages/python.md`
Integration test exclusion patterns (`tests/integration/`,
`**/test_integration.py`) are documented in both files. This is a maintenance
risk. **Fix:** remove the duplicate from the checklist and replace with a
cross-reference to `python.md`. (Same fix as P5-9.)

**P6-7** · `docs/recipe-handbook/anatomy.md`, line 61
`deployable` field description references "Agent Garden" with no definition
or link. If Agent Garden is an internal product, it should not appear in
public docs without explanation. **Fix:** either add a public link or rewrite
the description without the internal product name, e.g., "Set to `true` if
the recipe supports one-click deployment via the ADK toolchain."

---

## Persona 7 — Developer With a Failing PR

*I submitted my recipe. CI is red. I've pushed three fixes and it's still
failing. I'm not new to this — I know what ADK is, I know Python, I know uv.
I just need the docs to help me get unstuck fast.*

My first stop is `recipe-checklist.md` section 4, "When something fails." Two
bullets. "CI red? Go to troubleshooting." "Want the full story? Handbook." That
is the entirety of the guidance for someone whose PR is blocked. No list of
common failures. No hint about how to read the CI output. No "fix these in
this order." Just: go read another page. Not helpful.

I go to `troubleshooting.md`. No table of contents. I have to scroll 238 lines
to find my error. I'm copying the exact error text from the GitHub Actions log
and ctrl+F-ing on this page — and it finds nothing, because the section headers
here are paraphrased summaries, not the actual strings CI emits. I have to
read every header to find the one that matches my failure.

I finally find the right section. The fix for `env var used in source but
missing from .env.example` is: "Run `extract-python-environment-variables`."
That's an AI skill. I don't have an AI assistant configured right now. There
is no manual alternative anywhere on this page. I'm stuck on a fix that
requires a tool I can't use.

I push my fix. My PR is still red on a different check. Now I'm fixing things
one at a time, in whatever order I happen to find them, because there's no
guidance about which failures cascade from which. Turns out my `uv.lock` was
out of sync because I forgot to run `uv lock` after adding a dependency — and
that one root cause was also making `python-tests.yml` fail. I didn't know
that. The docs treat each error in isolation.

I'm also confused about the "Advisory notices" section. It looks exactly like
every other error section on the page — same heading level, same format. I
spent ten minutes trying to fix `Hardcoded model name` before I realized it
says "(never fail CI)" right there in the header. I missed it because I was
scanning, not reading.

After fixing everything I can find, I push again and wait. Does CI re-run
automatically? Do I need to push a new commit? Add a comment? The docs don't
say. I push an empty commit just to be safe, which is embarrassing and leaves
junk in my git history.

### Issues

**P7-1** · `docs/recipe-checklist.md`, section 4 — "When something fails" is inadequate
Two bullet points is not triage guidance. A developer in crisis needs more.
**Fix:** expand section 4 with at minimum: (1) a short list of the most
common first-submission failures in rough frequency order, (2) a note that
some failures cascade (e.g., fixing `uv.lock` may clear multiple checks at
once), and (3) the instruction to fix `validate-recipe-structure` failures
before Python-specific ones, since structural errors can mask downstream
checks.

**P7-2** · `docs/recipe-handbook/troubleshooting.md` — Section headers don't match CI error text
Every section header is a human-readable paraphrase of the failure. The actual
string in the GitHub Actions log may differ, so ctrl+F on the page often fails.
**Fix:** for each section, add the literal CI output string (or a close
substring) in a code block or as a secondary header line directly below the
section title. For example, under `## manifest.yaml missing / invalid per schema`,
add:
```
CI output contains: manifest.yaml not found
```
This gives readers a reliable ctrl+F target.

**P7-3** · `docs/recipe-handbook/troubleshooting.md` — No table of contents
Already flagged as P5-3. Reiterated here because this persona feels it most
acutely: they arrive at this page mid-crisis and cannot afford to scroll 238
lines. **Fix:** add a linked ToC at the top of `troubleshooting.md` (see
P5-3 for details). This is the single highest-impact change on this page.

**P7-4** · `docs/recipe-handbook/troubleshooting.md`, line 107 — AI-skill-only fix, no manual path
The fix for `env var used in source but missing from .env.example` is
"Run `extract-python-environment-variables`" — an AI skill. No manual
alternative is given. **Fix:** add a manual fallback below the skill reference:
"If you don't have an AI assistant, add the missing variable name to
`.env.example` manually. The CI error message will list the exact variable
name(s) that are missing."

**P7-5** · `docs/recipe-handbook/troubleshooting.md`, line 143 — AI-skill-only fix, no manual path
The fix for `tests/test_runnability.py missing` is "Run
`generate-python-runnability-test`" — an AI skill with no manual fallback.
**Fix:** add a reference to the copy-paste starter in `python.md`:
"Alternatively, copy the minimal template from
[python.md — tests/test_runnability.py](./languages/python.md#teststest_runnabilitypy)
and adjust it for your agent."

**P7-6** · `docs/recipe-handbook/troubleshooting.md` — Advisory notices look like blocking errors
The `## Advisory notices (never fail CI)` section uses the same `##` heading
level and identical formatting as every blocking error section. A developer
scanning quickly will miss "(never fail CI)" in the header and treat these as
blocking failures. **Fix:** (see also P5-8) visually distinguish this section
— use a `>` blockquote wrapper, a horizontal rule before it, or a lower
heading level — and add a bold preamble sentence: "**These will not block your
PR.** Fix them when convenient."

**P7-7** · `docs/recipe-handbook/troubleshooting.md` — "Something else" section buried at the bottom
The fallback for unlisted errors is the last section on a 238-line page. A
developer who can't find their error has to scroll to the very end to learn
what to do. **Fix:** add a short "Can't find your error?" note near the top
of the page (immediately after the opening sentence), linking to the
`## Something else` section. E.g., "Can't find your error? [Jump to the
bottom](#something-else) or search this page for keywords from the CI log."

**P7-8** · `docs/recipe-handbook/troubleshooting.md` — No guidance on re-triggering CI
After pushing a fix, it's not obvious whether CI re-runs automatically, or
whether a specific action is needed (empty commit, PR comment, re-run button).
**Fix:** add a one-paragraph "Re-running CI" note — either at the top of
`troubleshooting.md` or in checklist section 4 — explaining that CI triggers
on every push to the PR branch and that no manual re-trigger is normally
needed. Also note that if a check is stuck, reviewers can re-trigger via the
GitHub UI.

**P7-9** · `docs/recipe-handbook/troubleshooting.md` — Workflow names not linked
Each section starts with `**Workflow:** python-validate-recipe.yml` (or
similar) but the filename is plain text, not a link. A developer who wants to
read the actual workflow logic has to find the file themselves. **Fix:** link
each workflow name to its file in `.github/workflows/`, e.g.,
`[python-validate-recipe.yml](../../.github/workflows/python-validate-recipe.yml)`.

**P7-10** · `docs/recipe-handbook/troubleshooting.md`, line 112 — `check_env_vars.py` path missing
"File an issue against `check_env_vars.py`" names the file without a path.
A developer looking for it has to search the repo. **Fix:** provide the path:
`tools/check_env_vars.py` (verify actual path and update accordingly), and
link directly to the GitHub Issues page with the file path pre-filled or
noted in the issue template suggestion.

**P7-11** · `docs/recipe-handbook/troubleshooting.md` — Inconsistent fix format (prose vs. code blocks)
Some fixes have copy-pasteable code blocks (e.g., `uv.lock out of sync`,
`Missing package hash`, `Ruff format/check failed`). Many do not (e.g.,
`ownership.team is a placeholder`, `[project].name doesn't match folder name`,
`requires-python permits versions below 3.11`). A frustrated developer
copy-pasting fixes from this page encounters an inconsistent experience.
**Fix:** for every error whose fix is a shell command or a file edit, add a
fenced code block showing the exact command or the exact line to change. For
fixes that are purely conceptual (e.g., "rename the folder"), prose is
acceptable — but draw the line clearly.

---

## Cross-Cutting Issues

*Issues that appeared across multiple personas and affect the whole doc set.*

**X-1** · `docs/recipe-handbook/README.md`, line 37 — Broken anchor (affects P1, P3)
`./languages/python.md#new-python-recipe-from-scratch` resolves to nothing.
The heading in `python.md` is "The fast path" → anchor `#the-fast-path`.
**Fix:** update the link and its display text.

**X-2** · All docs — "poc" unexpanded (affects P1, P4)
`ownership.poc` is never expanded to "Point of Contact" anywhere in the docs.
**Fix:** add "(Point of Contact)" inline on first use in `anatomy.md`.

**X-3** · Multiple docs — "ship" as jargon (affects P4, P5)
"ship a recipe" is used in `recipe-checklist.md` and `skills-catalog.md`.
**Fix:** replace with "submit a recipe" throughout.

**X-4** · Multiple docs — 4-space vs fenced code blocks (affects P3, P5)
Inconsistent code formatting across all pages. **Fix:** standardize on fenced
``` blocks; convert all 4-space indented command examples.

**X-5** · `recipe-checklist.md` and `python.md` — Integration test duplication (affects P5, P6)
Two places document the same exclusion patterns. **Fix:** one source of truth
in `python.md`; a cross-reference in the checklist.
