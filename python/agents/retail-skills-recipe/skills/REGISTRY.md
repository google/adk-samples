# Skill Registry

| Skill | Version | Directory | Depends on |
|-------|---------|-----------|-----------|
| `retail-product-search` | 1.0 | [retail-product-search/](retail-product-search/) | -- |
| `retail-virtual-tryon` | 1.0 | [retail-virtual-tryon/](retail-virtual-tryon/) | retail-product-search |
| `retail-product-recommendation` | 1.0 | [retail-product-recommendation/](retail-product-recommendation/) | retail-product-search |
| `retail-content-generation` | 1.0 | [retail-content-generation/](retail-content-generation/) | retail-product-search |

## Dependency Graph

```
retail-product-search (base)
    +-- retail-virtual-tryon
    +-- retail-product-recommendation
    +-- retail-content-generation
```

## Installing a Skill

```bash
# From inside the cloned repo
node packages/install-vertical-skill/bin/install.js retail-virtual-tryon
node packages/install-vertical-skill/bin/install.js retail-virtual-tryon --target claude
```

The agent reads the SKILL.md and drives setup conversationally -- that is what skills are for.

## Adding a New Skill

1. Create `skills/{your-skill}/SKILL.md` -- follow the root `../SKILL.md` guide
2. Add `scripts/`, `assets/`, `tests/` inside the same skill directory
3. Create `skills/{your-skill}/EVAL.yaml` -- at least 5 eval cases
4. Add a row to this table
5. Open a PR -- CI will run `./vs eval {your-skill}` automatically
