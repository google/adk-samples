<!-- word count: 303 (target 500, cap 800) -->

# Recipe Handbook

You're here to contribute a recipe to `contrib/`. Welcome.

This handbook is the reference — standards, tooling, and the
*why* behind each. If you already know your way around, the
[checklist](../recipe-checklist.md) is your day-to-day tool.
You don't have to read every handbook page; the checklist
links back here whenever a step needs unpacking.

## What makes a good recipe

**Recipes must earn their place.** Have a clear intent, a
concrete problem the recipe solves for the ADK community, and
something new to teach. If you can't state it in one sentence,
revisit the idea before writing code. Recipes that duplicate
existing examples without new insight may not be accepted.

Every accepted recipe:

- Lives under `contrib/` with a valid `manifest.yaml` and a
  `README.md`.
- Passes the runnability test (agent code loads without
  crashing).
- Has real owners in `manifest.ownership`.

## Handbook pages

**If you're new here, start with:**

1. [Anatomy of a recipe](./anatomy.md) — what files every
   recipe has, and why
2. ["New Python recipe from scratch"](./scenarios.md#new-python-recipe-from-scratch)
   — end-to-end walk-through
3. [Python language rules](./languages/python.md) — the
   specific requirements

**Reference:**

- [Skills catalog](./skills-catalog.md) — the AI skills that
  do the work for you
- [Scenarios](./scenarios.md) — end-to-end walkthroughs
  beyond the first-recipe path
- [Troubleshooting](./troubleshooting.md) — errors mapped
  directly to fixes
- Other languages *(coming soon)*: Java · Go · TypeScript ·
  Kotlin

## Glossary

- **Recipe** — a runnable agent example (or importable agent
  module) under `contrib/`.
- **Skill** — a skill your AI assistant invokes (e.g.
  `prepare-python-recipe`). Lives in `.agents/skills/` and
  loads automatically.
- **Manifest** — `manifest.yaml`. Declares recipe metadata:
  type, language, ownership, description.

## Contact

Open a GitHub issue at
[github.com/google/adk-samples/issues](https://github.com/google/adk-samples/issues).
Include the recipe path and the CI check name if you're
reporting a failure.

---

← [Checklist](../recipe-checklist.md) · [Handbook](./README.md)
