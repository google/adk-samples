<!-- word count: 122 (target 300, cap 500) -->

# `contrib/`

Community-contributed ADK recipes. Each recipe is a
self-contained example covering a specific use case or industry
workflow.

## When to pick `contrib/`

- Community contribution, personal project, experiment, or
  demonstration of an idea.
- Faster review, lighter bar than any curated tier.

## Required files

Only the [universal set](../anatomy.md).

## Size limits

`contrib/` recipes should be self-contained and lean. Data
files, notebooks, and large assets belong in a linked storage
bucket, not the recipe. See
[anatomy.md#size-limits](../anatomy.md#size-limits) for the
numbers.

## Ownership

`manifest.ownership.team` and `manifest.ownership.poc` (Point of
Contact) are required — CI fails if they're placeholders. The
`poc` is notified when issues are opened about the recipe.

---

← [Checklist](../../recipe-checklist.md) · [Handbook](../README.md)
