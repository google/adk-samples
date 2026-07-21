<!-- word count: 371 (target 500, cap 800) -->

# Recipe Handbook

You're here to contribute a recipe to `contrib/`. Welcome.

This handbook explains the standards, the tooling, and the *why*
behind each. If you already know your way around, the
[checklist](../recipe-checklist.md) is your faster day-to-day
tool. **This page and the checklist together give you everything
you need — you don't have to read every handbook page.**

## What do you want to do?

Pick your path below. You can safely ignore the rest.

**Contributing your first recipe?**
[anatomy.md](./anatomy.md) shows the shape of a recipe. Then
walk through
["New Python recipe from scratch"](./workflows.md#new-python-recipe-from-scratch).

**Updating a recipe you already own?**
Re-run `prepare-python-recipe` — safe to re-run, applies
current standards automatically. See
["Updating an existing recipe"](./workflows.md#updating-an-existing-recipe)
for what happens under the hood.

**CI failing on your PR?**
[troubleshooting.md](./troubleshooting.md) — errors mapped
directly to fixes.

**Want the full picture?**
Read [anatomy.md](./anatomy.md), then
[ci-checks.md](./ci-checks.md). Together they cover the whole
system.

## Every recipe must earn its place

Have a clear intent, a concrete problem it solves for the ADK
community, and something new to teach. If you can't state it in
one sentence, revisit the idea before writing code. Recipes that
duplicate existing examples without new insight may not be
accepted.

## The recipe standard, at a glance

Every recipe:

- Lives under `contrib/` with a valid `manifest.yaml` and a
  `README.md`.
- Passes the runnability test (agent code loads without
  crashing).
- Has real owners in `manifest.ownership`.

## More detail if you need it

- [folders/contrib.md](./folders/contrib.md) — `contrib/`
  specifics
- [languages/python.md](./languages/python.md) — Python-specific
  rules (pyproject, .env.example, runnability test)
- [skills-catalog.md](./skills-catalog.md) — the AI skills that
  do the work for you
- [workflows.md](./workflows.md) — end-to-end scenarios
- [ci-checks.md](./ci-checks.md) — what each CI check enforces
- Other languages *(coming soon)*: Java · Go · TypeScript ·
  Kotlin

## Glossary

- **Recipe** — a runnable agent example (or importable agent
  module) under `contrib/`.
- **Skill** — a skill your AI assistant invokes (e.g.
  `prepare-python-recipe`). Lives in `.agents/skills/` and loads
  automatically.
- **Manifest** — `manifest.yaml`. Declares recipe metadata:
  type, language, ownership, description.

## Contact

Open a GitHub issue at
[github.com/google/adk-samples/issues](https://github.com/google/adk-samples/issues).
Include the recipe path and the CI check name if you're
reporting a failure.

---

← [Checklist](../recipe-checklist.md) · [Handbook](./README.md)
