# Drift checks

Two heavy CUJs live here: **does the repo still obey its own policy**, and **what
consumes this config key**. Both run only when the caller explicitly asks.

Report findings. Never fix them.

---

## Step zero: scope to the committed state

**Do this first or every finding is suspect.** The validators walk the working tree, so
they report the caller's local scratch directories as repo drift. A directory with an
ignored `.venv` and no `manifest.yaml` looks identical to a broken recipe.

```bash
git ls-files <path> | head -1        # empty output = untracked, not repo drift
```

Discard any finding whose path has no tracked files, or say explicitly that it is local
to the caller's machine. Telling an admin their repo is broken when it is that admin's
own scratch folder is the single worst failure this skill can produce.

---

## Check 1 — the repo's own validators

Fast enough for any question (well under a second on the current repo).

```bash
uv run validate all
```

Covers manifest schema, structure (required files and dirs, size, naming), README, and
placement. Scope it with `uv run validate all core` or a single recipe path.

Registered subcommands live in `tools/validate.py` under `SUBCOMMANDS`; run
`uv run validate` with no arguments for the current set rather than assuming this list.

**Note what placement does and does not cover.** It answers "is this recipe in the right
folder" for the roots it inspects. A recipe duplicated at an unexpected depth may pass
every check while still being wrong — compare `git ls-files '*/manifest.yaml'` against
the expected `<root>/<language|vertical>/<recipe>` shape when that is the question.

---

## Check 2 — the language list, across every source

The language set is duplicated in six places and nothing enforces agreement. This is
the highest-value consistency check in the repo.

```bash
uv run --with pyyaml python3 -c "
import json, re, yaml
p = yaml.safe_load(open('.github/policy.yml'))
srcs = {
  'schema.enum': set(json.load(open('.github/schemas/manifest-schema.json'))['properties']['language']['enum']),
  'policy.required_files': set(p['required_files']['by_language']),
  'policy.required_dirs': set(p['required_dirs']['by_language']),
  'policy.excluded_paths': set(p['excluded_paths']) - {'common'},
  'validate_manifest': set(re.search(r'LANGUAGE_NAMESPACE_DIRS = \{([^}]*)\}', open('tools/validate_manifest.py').read()).group(1).replace('\"','').replace(' ','').split(',')),
  'CODEOWNERS': set(re.findall(r'/(?:core|contrib)/([a-z]+)/', open('.github/CODEOWNERS').read())),
}
for lang in sorted(set().union(*srcs.values())):
    missing = [n for n, s in srcs.items() if lang not in s]
    print(f'{lang:12} {\"OK\" if not missing else \"MISSING FROM: \" + \", \".join(missing)}')
"
```

A language present in the schema but absent from `validate_manifest.py` is accepted by
schema validation and rejected by the validator — the confusing half-supported state the
add-a-language runbook exists to prevent. Dependabot ecosystems are deliberately not in
this check: the mapping from language to ecosystem is not one-to-one.

---

## Check 3 — values inlined instead of resolved

Any file that restates a number owned by `.github/policy.yml` will eventually contradict
it. Search docs and skills for policy-shaped claims:

```bash
grep -rnE "[0-9]+ ?(files|MB|days)" docs/ .agents/skills/ --include="*.md"
```

Review each hit: is it resolving the value from source, or asserting it? Known live
example — `.agents/skills/generate-manifest/SKILL.md` inlines the size limits and states
that `skills/` has none, while `policy.yml` defines them.

---

## Check 4 — dead pointers

Skills and docs route people to files by path. A renamed file turns a router into a dead
end silently, because only `docs/` links are covered by `docs-links.yml`.

```bash
grep -rhoE '`[.a-zA-Z0-9_/-]+\.(md|py|ya?ml|json)`' .agents/skills/ --include="*.md" \
  | tr -d '`' | sort -u | while read -r f; do [ -e "$f" ] || echo "MISSING: $f"; done
```

Paths inside fenced examples produce false positives; confirm before reporting.

---

## Check 5 — catalogue versus reality

```bash
ls -1 .agents/skills/
grep -oE '^### `[a-z-]+`' docs/recipe-handbook/skills-catalog.md
```

A skill on disk but absent from the catalogue is invisible to contributors. A catalogue
entry with no skill is a broken promise. Neither is CI-enforced.

---

## Check 6 — ownership that has gone stale

```bash
grep -rn "TODO: Replace with your" --include="manifest.yaml" core/ contrib/ skills/
```

`tools/validate_manifest.py` already fails on the literal placeholder strings, so hits
here mean a recipe that cannot be passing validation. Separately, a `poc` naming someone
who has left is invisible to every check — worth reporting when the caller asks who owns
something and the answer looks wrong.

---

## Check 7 — CODEOWNERS pointing at nothing

```bash
grep -oE '^/[^ ]+' .github/CODEOWNERS | sed 's|/\*\*$||; s|^/||' \
  | while read -r p; do [ -e "$p" ] || echo "NO SUCH PATH: $p"; done
```

A rule for a directory that does not exist is harmless but misleading — it reads as
though the language is supported.

---

## Tracing what consumes a config key

The blast-radius question. Given a key such as `deployability.min_google_adk`:

1. **Search the dotted path**, for consumers reading it through `load_policy.py`:
   ```bash
   grep -rn "deployability.min_google_adk" . --exclude-dir=.git --exclude-dir=.venv
   ```
2. **Search the leaf key name alone**, for code that loads the YAML and subscripts it:
   ```bash
   grep -rn "min_google_adk" . --exclude-dir=.git --exclude-dir=.venv
   ```
3. **Search the literal value**, which is what finds the dangerous case — somewhere that
   hardcoded the number instead of reading it. These are the hits that turn "edit one
   line" into a multi-file change.
4. **Check the four consumer families**: `.github/workflows/`, `.github/scripts/`,
   `tools/`, and `.agents/skills/` — plus `docs/`, which should link rather than restate.

**Never filter the search by file extension.** `.github/CODEOWNERS` has none, and an
extension-filtered search silently omits it. That mistake was made while building this
skill.

Report consumers grouped by family, and say plainly whether the change is confined to
`policy.yml` or reaches code.
