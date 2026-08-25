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
git ls-files '<path>' | head -1        # empty output = untracked, not repo drift
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

The language set is duplicated across every source the script below reads, and nothing
enforces agreement. This is the highest-value consistency check in the repo.

```bash
uv run --with pyyaml python3 -c "
import json, re, yaml
policy = yaml.safe_load(open('.github/policy.yml'))
schema = json.load(open('.github/schemas/manifest-schema.json'))
validate_manifest_src = open('tools/validate_manifest.py').read()
codeowners_src = open('.github/CODEOWNERS').read()

# Keys are lowercase dotted 'file.thing' paths, so the MISSING FROM line below
# names where to go and not just what is wrong.
sources = {
  'schema.language_enum': set(schema['properties']['language']['enum']),
  'policy.required_files': set(policy['required_files']['by_language']),
  'policy.required_dirs': set(policy['required_dirs']['by_language']),
  'policy.excluded_paths': set(policy['excluded_paths']) - {'common'},
  # findall over the quoted strings, not a comma split: the set may wrap across
  # lines or carry a trailing comma, either of which leaves newlines and an
  # empty string among the names and makes every comparison miss.
  'validate_manifest.language_namespace_dirs': set(re.findall(r'[\"\\']([a-z]+)[\"\\']', re.search(r'LANGUAGE_NAMESPACE_DIRS\s*=\s*\{([^}]*)\}', validate_manifest_src).group(1))),
  'codeowners.recipe_roots': set(re.findall(r'/(?:core|contrib)/([a-z]+)/', codeowners_src)),
}
for language in sorted(set().union(*sources.values())):
    absent_from = [name for name, langs in sources.items() if language not in langs]
    print(f'{language:12} {\"OK\" if not absent_from else \"MISSING FROM: \" + \", \".join(absent_from)}')
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

A path is resolved **relative to the file that names it**, then relative to the repo
root — `SKILL.md` writes `reference/runbooks.md` meaning its own sibling, and testing
that from the root reports every one of them missing.

The pattern requires a `/`, so a bare `SKILL.md` or `__init__.py` — named as a kind of
file rather than as a pointer to one — does not become a finding.

```bash
grep -rnoE '`[.a-zA-Z0-9_-]+(/[.a-zA-Z0-9_-]+)+\.(md|py|ya?ml|json)`' \
  .agents/skills/ --include="*.md" \
  | tr -d '`' | while IFS=: read -r src _ f; do
      skill_root="$(printf '%s' "$src" | cut -d/ -f1-3)"
      [ -e "$f" ] || [ -e "$skill_root/$f" ] || [ -e "$(dirname "$src")/$f" ] \
        || echo "MISSING: $f  (named in $src)"
    done | sort -u
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

CODEOWNERS patterns contain globs, so the existence test has to expand them. Quoting
`"$p"` makes the shell look for a literal `*` and report every wildcard rule as
missing.

```bash
grep -oE '^/[^ ]+' .github/CODEOWNERS | sed 's|/\*\*$||; s|^/||' \
  | while read -r p; do
      # Unquoted on purpose: this is the expansion. `nullglob` makes a pattern
      # that matches nothing expand to zero words rather than to itself.
      #
      # BOTH tests are needed. A path with no wildcard is never expanded, so
      # it survives as one literal word and `$#` is 1 whether or not it
      # exists. Counting alone therefore passes every non-glob path — which
      # is most of the file — and the check silently finds nothing.
      ( shopt -s nullglob; set -- $p; [ "$#" -gt 0 ] && [ -e "$1" ] ) \
        || echo "NO SUCH PATH: $p"
    done
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
4. **Check every consumer family**: `.github/workflows/`, `.github/scripts/`,
   `tools/`, and `.agents/skills/` — plus `docs/`, which should link rather than restate.

**Never filter the search by file extension.** `.github/CODEOWNERS` has none, and an
extension-filtered search silently omits it. That mistake was made while building this
skill.

Report consumers grouped by family, and say plainly whether the change is confined to
`policy.yml` or reaches code.
