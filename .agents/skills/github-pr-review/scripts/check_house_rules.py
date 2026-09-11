#!/usr/bin/env python3
"""Check a google/adk-samples recipe against the house rules in .github/review-rules.md.

    check_house_rules.py --repo-root /path/to/checkout --recipe contrib/python/foo
    check_house_rules.py --repo-root . --recipe contrib/python/foo --json

Emits findings in the same schema the analysis lanes use, with `rule`, `ci` and
`anchorable` pre-populated. Deterministic: no model, no judgement, no tokens.

Covers the 21 mechanically-decidable rules. Four are deliberately NOT here because
they need an AST or a judgement call, and a wrong "this fails CI" costs more than a
missed nit:

    H11  hardcoded model literal      — four AST exemptions (collection literal,
                                        subscript, comparison operand, docstring)
    H12  default on a model-name var  — requires knowing which var is a model var
    H16  noqa E402 on trailing import — needs "after first non-import statement"
    H25  runnability assert placement — needs the `with patch(...)` block structure

Three rules here have traps that a naive grep gets wrong, and each is called out at
its implementation:

    H9   `source = { editable = "." }` is the recipe's OWN package, not a violation
    H15  needs package-root resolution among many __init__.py, and is a NEGATIVE
    H19  needs the real JSON schema; `license`/`tags` ARE permitted keys
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # py<3.11
    import tomli as tomllib

CI_FAIL = "fail"
CI_ADV = "advisory"

# Rules that could not be evaluated this run. Always reported -- a silent skip
# makes every "no violations" result a lie.
SKIPPED = []


def _parse_manifest(text):
    """(data, degraded). Falls back to a top-level-key scan without pyyaml."""
    try:
        import yaml
        return (yaml.safe_load(text) or {}), False
    except ModuleNotFoundError:
        pass
    except Exception:
        return None, False
    data = {}
    for line in text.split("\n"):
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$', line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if v.startswith(('"', "'")) and v.endswith(('"', "'")) and len(v) > 1:
            v = v[1:-1]
        data[k] = v if v else {}
    return data, True


# Rules that describe the recipe as a whole rather than a specific file. They are
# only the PR's responsibility when the PR is adding the recipe.
WHOLE_RECIPE_RULES = {"H21", "H22", "H23"}
# H24 is the exception: modifying ANY file under a frozen path is itself the
# violation, so a small edit to a legacy recipe does trigger it.
# H48 is the other: a junk ownership.team is the one thing nobody else catches
# (the schema only asks for a non-empty string), and letting it through once is
# what produced a fleet of recipes owned by "Google". It is always reported;
# when manifest.yaml is outside the diff the finding lands in the un-anchorable
# bucket rather than being dropped.
ALWAYS_RULES = {"H24", "H48"}

CHANGED = None          # set from --changed-files; None means "audit everything"
NEW_RECIPE = True       # set False when the PR only edits an existing recipe
FILTERED = []           # rules suppressed as pre-existing, for reporting


def _is_ours(rule, path):
    """Is this violation something the PR introduced, or pre-existing noise?"""
    if CHANGED is None:
        return True
    if rule in ALWAYS_RULES:
        return True
    if rule in WHOLE_RECIPE_RULES:
        return NEW_RECIPE
    return path in CHANGED


def fetch_changed_files(repo, pr):
    """Every changed path, paged.

    `gh pr view --json files` silently caps at 100. On PR #2302 (787 files) that
    truncation left pyproject.toml out of the list, so every finding in it was
    filtered away as "pre-existing" on a brand-new recipe.
    """
    out, page = set(), 1
    while True:
        proc = subprocess.run(
            ["gh", "api", f"/repos/{repo}/pulls/{pr}/files?per_page=100&page={page}"],
            capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise SystemExit(f"gh api failed: {proc.stderr.strip()[:200]}")
        batch = json.loads(proc.stdout or "[]")
        if not batch:
            break
        out.update(f["filename"] for f in batch)
        if len(batch) < 100:
            break
        page += 1
    return out



def find(out, rule, ci, path, line, what, evidence, verify):
    if not _is_ours(rule, path):
        FILTERED.append((rule, path))
        return
    out.append({
        "rule": rule, "ci": ci, "path": path, "line": line,
        "severity": "no_critical", "confidence": "high", "cheap": "cheap",
        "what": what, "evidence": evidence, "verify_steps": verify,
        "window": "", "unchecked": None,
    })


def read(p):
    try:
        return open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return None


def lineno_of(text, pattern):
    for i, l in enumerate(text.split("\n"), 1):
        if re.search(pattern, l):
            return i
    return 1


def load_toml(p):
    try:
        with open(p, "rb") as fh:
            return tomllib.load(fh)
    except Exception:
        return None


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------- H26 / H27

APACHE_HEADER = [
    "Copyright {year} Google LLC",
    "",
    'Licensed under the Apache License, Version 2.0 (the "License");',
    "you may not use this file except in compliance with the License.",
    "You may obtain a copy of the License at",
    "",
    "    https://www.apache.org/licenses/LICENSE-2.0",
    "",
    "Unless required by applicable law or agreed to in writing, software",
    'distributed under the License is distributed on an "AS IS" BASIS,',
    "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
    "See the License for the specific language governing permissions and",
    "limitations under the License.",
]
HEADER_EXT = {".py": "#", ".sh": "#", ".tf": "#", ".yaml": "#", ".yml": "#",
              ".ts": "//", ".tsx": "//", ".js": "//", ".jsx": "//"}
# Source files are EXPECTED to carry the header, so a missing one is a violation
# however many others also lack it. Config formats only have to be self-consistent
# -- if no .tf in the recipe has a header, that is the convention there.
HEADER_REQUIRED = {".py", ".ts", ".tsx", ".js", ".jsx", ".sh"}
# Blank separator lines carry no text, so only the prose lines are matchable.
# Deriving this rather than hardcoding it: a literal 10 was unreachable against
# the 9 real lines, so every file scored "partial" and the check silently died.
_HEADER_LINES = [w for w in APACHE_HEADER[1:] if w]
_FULL_AT = len(_HEADER_LINES) - 1          # tolerate one reflowed line
_ENV_READERS = {"getenv", "environ"}


def _env_read_defaults(src, path):
    """[(line, call_text, why)] for env reads that carry a hardcoded default.

    AST, not regex: a regex flags the same call inside a docstring or a comment,
    and cannot tell `os.environ["X"]` (fine -- cannot carry a default) from
    `os.environ.get("X", "d")` (not fine).
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    def env_call(node):
        """('getenv'|'get'|'setdefault') if this Call reads the environment."""
        fn = node.func
        if not isinstance(fn, ast.Attribute):
            return None
        if fn.attr == "getenv" and isinstance(fn.value, ast.Name) and fn.value.id == "os":
            return "getenv"
        if fn.attr in ("get", "setdefault"):
            v = fn.value
            if isinstance(v, ast.Attribute) and v.attr == "environ":
                return fn.attr
            if isinstance(v, ast.Name) and v.id == "environ":
                return fn.attr
        return None

    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            kind = env_call(node)
            if not kind:
                continue
            if len(node.args) >= 2 or any(k.arg == "default" for k in node.keywords):
                out.append((node.lineno, kind,
                            "second argument is a hardcoded default"))
        # `os.getenv("X") or "fallback"` -- semantically a default.
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            first = node.values[0]
            if isinstance(first, ast.Call) and env_call(first):
                rest = node.values[1:]
                if any(isinstance(v, ast.Constant) and v.value not in (None, "")
                       for v in rest):
                    out.append((node.lineno, "or",
                                "`or` fallback after an env read"))
    return out


def check_env_defaults(out, root, rel):
    """H26 -- an env read must not carry a hardcoded default."""
    recipe_abs = os.path.join(root, rel)
    hits = []
    for dirpath, dirnames, filenames in os.walk(recipe_abs):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".venv", "node_modules", ".git")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(dirpath, fn)
            relp = os.path.relpath(fp, root)
            if CHANGED is not None and relp not in CHANGED:
                continue
            src = read(fp)
            if not src:
                continue
            for line, kind, why in _env_read_defaults(src, relp):
                hits.append((relp, line, kind, why))
    if not hits:
        return
    # One finding, not one per hit -- the grouping rule.
    relp, line, kind, why = hits[0]
    others = sorted({h[0] for h in hits})
    find(out, "H26", CI_ADV, relp, line,
         f"env read carries a hardcoded default ({why}); "
         f"{len(hits)} occurrence(s) across {len(others)} file(s). Defaults belong "
         "in .env.example, not in the code",
         "; ".join(f"{h[0]}:{h[1]}" for h in hits[:6]),
         f"read line {line}; the call has a second argument")


def _header_state(text, marker):
    """('full'|'partial'|'none', matched_lines)."""
    lines = [l.rstrip() for l in text.split("\n")[:25]]
    body = []
    for l in lines:
        s = l.strip()
        if s.startswith(marker):
            body.append(s[len(marker):].strip())
        elif not s and body:
            body.append("")
        elif body:
            break
    if not body:
        return "none", 0
    joined = " ".join(body)
    matched = sum(1 for want in _HEADER_LINES
                  if want.strip()[:40] in joined)
    has_copyright = "Copyright" in joined and "Google LLC" in joined
    if matched >= _FULL_AT:
        return "full", matched
    if has_copyright or matched:
        return "partial", matched
    return "none", 0


def check_license_headers(out, root, rel):
    """H27 -- the licence header must be consistent WITHIN each file type.

    Compared per extension, not pooled. Every .tf file in this recipe lacks a
    header; that is the convention for terraform here, not 12 violations. Pooling
    them against 662 headed .py files produced exactly that false positive.
    """
    recipe_abs = os.path.join(root, rel)
    by_ext = {}
    for dirpath, dirnames, filenames in os.walk(recipe_abs):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".venv", "node_modules", ".git",
                                    "dist", "build", ".next")]
        for fn in filenames:
            ext = os.path.splitext(fn)[1]
            marker = HEADER_EXT.get(ext)
            if not marker:
                continue
            fp = os.path.join(dirpath, fn)
            relp = os.path.relpath(fp, root)
            if CHANGED is not None and relp not in CHANGED:
                continue
            if fn.endswith((".gen.ts", ".gen.tsx")) or "generated" in relp:
                continue
            text = read(fp)
            if text is None or not text.strip():
                continue
            state, _ = _header_state(text, marker)
            by_ext.setdefault(ext, {"full": [], "partial": [], "none": []})
            by_ext[ext][state].append(relp)

    for ext, g in sorted(by_ext.items()):
        total = sum(len(v) for v in g.values())
        full = len(g["full"])
        if ext in HEADER_REQUIRED:
            if total < 2:
                continue                    # a lone file establishes nothing
        else:
            # Config formats: consistency only. No convention, no finding.
            if total < 5 or full == 0 or full / total < 0.6:
                continue
        dominant_full = full / total >= 0.6
        if g["partial"] or g["none"]:
            if dominant_full:
                # A convention exists and a minority breaks it.
                if g["partial"]:
                    find(out, "H27", CI_ADV, g["partial"][0], 1,
                         f"licence header is truncated here; {len(g['partial'])} "
                         f"{ext} file(s) differ from the {full} carrying the full "
                         "Apache block",
                         "; ".join(g["partial"][:6]),
                         f"compare the first 15 lines of {g['partial'][0]} "
                         "with a sibling")
                if g["none"]:
                    find(out, "H27", CI_ADV, g["none"][0], 1,
                         f"no licence header on this file; {len(g['none'])} {ext} "
                         f"file(s) have none while {full} carry the full block",
                         "; ".join(g["none"][:6]),
                         f"read the first 15 lines of {g['none'][0]}")
            elif full:
                # Two competing shapes. Say that, rather than implying the
                # majority is a broken version of the minority.
                find(out, "H27", CI_ADV, g["partial"][0] if g["partial"]
                     else g["none"][0], 1,
                     f"the {ext} files carry two different headers: "
                     f"{len(g['partial'])} a shorter notice, {len(g['none'])} none, "
                     f"and {full} the full Apache block",
                     "; ".join((g["partial"] + g["none"])[:6]),
                     f"compare the first 15 lines of {(g['partial'] or g['none'])[0]} "
                     f"against {g['full'][0]}")
            else:
                find(out, "H27", CI_ADV, (g["partial"] or g["none"])[0], 1,
                     f"no {ext} file in this recipe carries the standard Apache "
                     f"header ({len(g['partial'])} have a shorter notice, "
                     f"{len(g['none'])} have none)",
                     "; ".join((g["partial"] + g["none"])[:6]),
                     f"read the first 15 lines of {(g['partial'] or g['none'])[0]}")


def check_pyproject(out, root, rel, recipe_name):
    p = os.path.join(root, rel, "pyproject.toml")
    text = read(p)
    if text is None:
        return
    r = os.path.join(rel, "pyproject.toml")
    data = load_toml(p) or {}
    proj = data.get("project", {})

    # H1 -- ruff config belongs to the repo root only
    hits = [i for i, l in enumerate(text.split("\n"), 1)
            if re.match(r"^\s*\[tool\.ruff(\.|\])", l)]
    if hits:
        find(out, "H1", CI_FAIL, r, hits[0],
             f"declares {len(hits)} [tool.ruff*] table(s); recipes must not "
             f"(lines {', '.join(map(str, hits))})",
             "AGENTS.md:50-52; python-validate-recipe.yml:261-266",
             "grep '^\\[tool\\.ruff' in this file")

    # H3 -- [project].name must equal the folder basename
    name = proj.get("name")
    if name and name != recipe_name:
        find(out, "H3", CI_FAIL, r, lineno_of(text, r'^\s*name\s*='),
             f'[project].name is "{name}" but the folder is "{recipe_name}"',
             "check_recipe_pyproject.py:92-114",
             "compare name= against the directory basename")

    # H4 -- must accept 3.11 exactly. Specifier logic, not a string match.
    rp = proj.get("requires-python")
    ln = lineno_of(text, r'^\s*requires-python\s*=')
    if not rp:
        find(out, "H4", CI_FAIL, r, 1, "no requires-python declared",
             "check_recipe_pyproject.py:117-197", "read [project]")
    else:
        bad = None
        for part in [s.strip() for s in rp.split(",")]:
            m = re.match(r"(>=|>|~=|==)\s*3\.(\d+)", part)
            if not m:
                continue
            op, minor = m.group(1), int(m.group(2))
            if op in (">=", "~=", "==") and minor < 11:
                bad = f"{rp} permits Python below 3.11"
            if op in (">=", "~=", "==") and minor > 11:
                bad = f"{rp} excludes Python 3.11 (CI pins 3.11 for uv lock --check)"
        if bad:
            find(out, "H4", CI_FAIL, r, ln, bad,
                 "check_recipe_pyproject.py:117-197", "read the requires-python specifier")

    # H5 -- [[tool.uv.index]] array-of-tables, default=true, public PyPI
    idx = data.get("tool", {}).get("uv", {}).get("index")
    ok_urls = {"https://pypi.org/simple", "https://pypi.org/simple/"}
    if idx is None:
        find(out, "H5", CI_FAIL, r, 1, "no [[tool.uv.index]] declared",
             "check_recipe_pyproject.py:262-340", "read [tool.uv]")
    elif isinstance(idx, dict):
        # single-bracket [tool.uv.index] instead of array-of-tables
        find(out, "H5", CI_FAIL, r, lineno_of(text, r"tool\.uv\.index"),
             "[tool.uv.index] must be an array-of-tables [[tool.uv.index]]",
             "check_recipe_pyproject.py:298-308", "check the bracket count")
    else:
        defaults = [e for e in idx if e.get("default") is True]
        if not defaults:
            find(out, "H5", CI_FAIL, r, lineno_of(text, r"tool\.uv\.index"),
                 "no [[tool.uv.index]] entry has default = true",
                 "check_recipe_pyproject.py:262-340", "read the index entries")
        elif defaults[0].get("url") not in ok_urls:
            find(out, "H5", CI_FAIL, r, lineno_of(text, r"tool\.uv\.index"),
                 f'default index is "{defaults[0].get("url")}", must be public PyPI',
                 "check_recipe_pyproject.py:262-340", "read the default index url")

    # H6 -- python-dotenv in [project].dependencies (dev group does NOT count)
    deps = proj.get("dependencies", []) or []
    names = {re.match(r"^\s*([A-Za-z0-9_.\-]+)", d).group(1).lower()
             for d in deps if re.match(r"^\s*([A-Za-z0-9_.\-]+)", d)}
    if "python-dotenv" not in names:
        find(out, "H6", CI_ADV, r, lineno_of(text, r"^\s*dependencies\s*="),
             "python-dotenv is not in [project].dependencies "
             "(a dev dependency-group does not count)",
             "extract_env_vars.py:1797-1839", "read [project].dependencies")

    # H7 -- build-system with both keys
    bs = data.get("build-system", {})
    if not bs or not bs.get("requires") or not bs.get("build-backend"):
        find(out, "H7", CI_ADV, r, lineno_of(text, r"\[build-system\]"),
             "[build-system] missing or lacks requires / build-backend",
             "align_pyproject.py:667", "read [build-system]")

    # H8 -- testpaths, if present, must collect the runnability test
    tp = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("testpaths")
    if tp:
        entries = [tp] if isinstance(tp, str) else list(tp)
        ok = {"", ".", "tests", "tests/test_runnability.py"}
        if not any(e.strip().rstrip("/") in ok for e in entries):
            find(out, "H8", CI_ADV, r, lineno_of(text, r"testpaths"),
                 f"testpaths {entries} never collects tests/test_runnability.py",
                 "align_pyproject.py:1063-1121", "read testpaths")


def check_uv_lock(out, root, rel, recipe_name, pyproject_name):
    """H9. TRAP: `source = { editable = "." }` is the recipe's OWN package."""
    p = os.path.join(root, rel, "uv.lock")
    text = read(p)
    if text is None:
        return
    r = os.path.join(rel, "uv.lock")
    own = {pyproject_name, recipe_name, recipe_name.replace("-", "_")}
    pkg = None
    for i, line in enumerate(text.split("\n"), 1):
        m = re.match(r'^\s*name\s*=\s*"([^"]+)"', line)
        if m:
            pkg = m.group(1)
            continue
        m = re.match(r"^\s*source\s*=\s*\{\s*(\w+)\s*=", line)
        if m and m.group(1) in ("git", "editable", "directory"):
            if m.group(1) in ("editable", "directory") and pkg in own:
                continue          # the recipe self-reference -- correct, not a defect
            find(out, "H9", CI_FAIL, r, i,
                 f'package "{pkg}" uses a {m.group(1)} source; '
                 "recipes must depend only on published PyPI releases",
                 "python-dependency-policy.yml:224-303", f"read line {i}")


def check_dotenv_bootstrap(out, root, rel):
    """H15. TRAP: needs package-root resolution, and asserts a NEGATIVE.

    The package root is the directory holding __init__.py that is a *direct child*
    of the recipe root -- not any of the nested ones, and not tests/ or eval/.
    """
    recipe_abs = os.path.join(root, rel)
    pkg_dirs = []
    try:
        for entry in sorted(os.listdir(recipe_abs)):
            d = os.path.join(recipe_abs, entry)
            if (os.path.isdir(d) and entry not in ("tests", "eval", "scripts", "docs")
                    and not entry.startswith(".")
                    and os.path.exists(os.path.join(d, "__init__.py"))):
                pkg_dirs.append(entry)
    except OSError:
        return
    if not pkg_dirs:
        return                      # no package -- H15 does not apply
    pkg = pkg_dirs[0]
    init_rel = os.path.join(rel, pkg, "__init__.py")
    init_text = read(os.path.join(root, init_rel)) or ""
    if "load_dotenv" in init_text:
        return                      # compliant

    # Where IS it called? Reporting the absence alone is what got this wrong before.
    callers = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(recipe_abs, pkg)):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".venv")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(dirpath, fn)
            t = read(fp) or ""
            if re.search(r"^\s*load_dotenv\s*\(", t, re.M):
                callers.append(os.path.relpath(fp, recipe_abs))
    if not callers:
        return                      # reads no dotenv at all -- not this rule's business
    find(out, "H15", CI_ADV, init_rel, 1,
         f"load_dotenv() is called from {len(callers)} module(s) "
         f"({', '.join(sorted(callers)[:4])}) but not from the package __init__.py",
         "docs/recipe-handbook/languages/python.md:107-110",
         f"grep load_dotenv in {pkg}/ and check {pkg}/__init__.py")


# ------------------------------------------------------------------------ H48

# Values that name no owner. Normalised (lowercased, punctuation and a trailing
# "team"/"group" stripped) before lookup, so "Google, LLC" and "ADK Samples Team"
# both land here. Deliberately a closed list: an unfamiliar name like "OpenEAGO"
# or "attenu-io" is somebody's real org, and guessing "that looks like a handle"
# from its shape flags those far more often than it flags a cheat.
GENERIC_TEAMS = {
    "google", "google llc", "google inc", "google cloud", "google cloud platform",
    "googler", "googlers", "gcp", "cloud", "alphabet",
    "adk", "adk samples", "adk-samples", "adk sample", "samples", "sample",
    "recipes", "recipe", "contrib", "community", "open source", "opensource",
    "team", "the", "my", "our", "n/a", "na", "none", "null", "nil", "unknown",
    "tbd", "todo", "xxx", "placeholder", "test", "testing", "demo", "example",
    "self", "me", "myself", "personal", "individual", "independent", "solo",
    "eng", "engineering", "dev", "devs", "developer", "developers", "devrel",
    "misc", "other", "others", "internal", "external", "public",
}


def _norm_team(value):
    """Lowercase, de-punctuate, and drop a trailing team/group/org noun."""
    v = re.sub(r"[^a-z0-9&/ -]", " ", str(value).lower())
    v = re.sub(r"\s+", " ", v).strip(" -&/")
    stripped = re.sub(
        r"\b(team|group|org|organisation|organization|llc|inc)\b\s*$",
        "", v).strip(" -&/")
    # "Team" on its own strips to nothing; keep the original so it still matches.
    return re.sub(r"\s+", " ", stripped or v)


def _ownership(data, text):
    """(team, poc, [contributors]) with a regex fallback for degraded parses."""
    own = data.get("ownership")
    if isinstance(own, dict) and own:
        contrib = own.get("contributors")
        return (own.get("team"), own.get("poc"),
                contrib if isinstance(contrib, list) else [])

    # Degraded parse (no pyyaml): the top-level scan yields `ownership: {}`, so
    # read the two scalars off the indented lines. A trailing `# comment` is part
    # of the line and not part of the value.
    def scalar(key):
        m = re.search(rf"^\s+{key}:[ \t]*(.+?)[ \t]*$", text, re.M)
        if not m:
            return None
        v = m.group(1)
        q = re.match(r"""^(['"])(.*?)\1""", v)
        v = q.group(2) if q else v.split("#")[0].strip()
        return v.strip() or None

    return scalar("team"), scalar("poc"), []


def check_ownership_team(out, r, text, data):
    """H48 -- ownership.team must name a team, not an org or a person.

    The schema asks only for minLength 1, so "Google" and the author's own GitHub
    handle both sail through every deterministic check in the repo. Neither tells
    a future maintainer who to page.
    """
    team, poc, contributors = _ownership(data, text)
    if not team or not isinstance(team, str):
        return                      # absent/typed wrong -- H19 and the schema
    raw = team.strip()
    if not raw:
        return
    # A live placeholder is H17's finding; two comments on one line is noise.
    if raw.upper().startswith("TODO") or "TODO:" in raw:
        return

    line = lineno_of(text, r"^\s+team:")
    norm = _norm_team(raw)
    handles = {str(h).strip().lstrip("@").lower()
               for h in ([poc] + list(contributors)) if h}
    verify = f"read the ownership.team value at line {line}"

    if norm in GENERIC_TEAMS:
        find(out, "H48", CI_ADV, r, line,
             f'ownership.team is "{raw}" -- that names an organisation, not a '
             "team. It must be the team that will maintain the recipe, specific "
             "enough that someone can find them",
             ".github/schemas/manifest-schema.json (ownership.team)", verify)
        return

    if raw.strip().lstrip("@").lower() in handles:
        who = "poc" if str(poc or "").strip().lstrip("@").lower() == \
            raw.strip().lstrip("@").lower() else "a contributor"
        find(out, "H48", CI_ADV, r, line,
             f'ownership.team is "{raw}", the same GitHub handle as {who}. '
             "team is the owning team; a personal handle leaves the recipe "
             "unowned the moment that person moves",
             ".github/schemas/manifest-schema.json (ownership.team)", verify)
        return

    if re.search(r"https?://|github\.com/|\S+@\S+\.\S+", raw):
        find(out, "H48", CI_ADV, r, line,
             f'ownership.team is "{raw}" -- a URL or an address, not a team name',
             ".github/schemas/manifest-schema.json (ownership.team)", verify)
        return

    if len(raw) < 2 or re.fullmatch(r"[^A-Za-z0-9]+", raw):
        find(out, "H48", CI_ADV, r, line,
             f'ownership.team is "{raw}", which names nobody',
             ".github/schemas/manifest-schema.json (ownership.team)", verify)


def check_manifest(out, root, rel, schema_path):
    p = os.path.join(root, rel, "manifest.yaml")
    text = read(p)
    if text is None:
        return
    r = os.path.join(rel, "manifest.yaml")

    # H17 -- the two canonical placeholders. Never suggest a value for these.
    for ph in ("TODO: Replace with your team name",
               "TODO: Replace with your GitHub user ID"):
        if ph in text:
            find(out, "H17", CI_FAIL, r, lineno_of(text, re.escape(ph)),
                 f'ownership placeholder still present: "{ph}"',
                 "validate_manifest.py:140-151", "grep the literal string")

    # A manifest is flat enough that a missing pyyaml must not silently disable
    # H18/H19 -- a checker that quietly skips rules is worse than no checker.
    data, degraded = _parse_manifest(text)
    if data is None:
        SKIPPED.append(("H18/H19", "could not parse manifest.yaml"))
        return
    if degraded:
        SKIPPED.append(("H19-nested", "pyyaml missing; only top-level keys checked"))

    # H48 -- ownership.team. Runs before H18 so the ownership comment is first
    # in the manifest's findings.
    check_ownership_team(out, r, text, data)

    # H18 -- description
    desc = (data.get("description") or "").strip()
    if desc.upper().startswith("TODO") or len(desc) < 10:
        find(out, "H18", CI_FAIL, r, lineno_of(text, r"^description:"),
             "description is a TODO placeholder or shorter than 10 characters",
             "validate_manifest.py:159-166", "read the description value")

    # H19 -- schema keys/enums. TRAP: license/tags/deployable/large ARE permitted.
    schema = None
    if schema_path and os.path.exists(schema_path):
        try:
            schema = json.load(open(schema_path))
        except Exception:
            schema = None
    if schema:
        allowed = set(schema.get("properties", {}))
        for k in data:
            if k not in allowed:
                find(out, "H19", CI_FAIL, r, lineno_of(text, rf"^{re.escape(k)}:"),
                     f'"{k}" is not a key in manifest-schema.json',
                     "manifest-schema.json (additionalProperties: false)",
                     "compare keys against the schema")
        for k, spec in schema.get("properties", {}).items():
            if k in data and "enum" in spec and data[k] not in spec["enum"]:
                find(out, "H19", CI_FAIL, r, lineno_of(text, rf"^{re.escape(k)}:"),
                     f'{k} = "{data[k]}" is not one of {spec["enum"]}',
                     "manifest-schema.json", "compare against the enum")


def check_readme(out, root, rel):
    p = os.path.join(root, rel, "README.md")
    text = read(p)
    if text is None:
        return
    r = os.path.join(rel, "README.md")
    words = len(text.split())
    if words < 100:
        find(out, "H20", CI_FAIL, r, 1, f"README is {words} words, minimum is 100",
             "validate_readme.py:42", "word-count the file")
    if "TODO:" in text:
        find(out, "H20", CI_FAIL, r, lineno_of(text, r"TODO:"),
             "README still contains TODO: text",
             "validate_readme.py:108", "grep TODO:")
    if not re.search(r"^```", text, re.M):
        find(out, "H20", CI_FAIL, r, 1, "README has no fenced code block",
             "validate_readme.py:132", "grep for a ``` fence")
    heads = re.findall(r"^#+ .*$", text, re.M)
    setup = r"setup|prerequisit|installation|install|requirement|configuration|getting started|before you begin|environment"
    run = r"\brun\b|running|usage|quickstart|quick start|\bstart\b|deploy|launch"
    if not any(re.search(setup, h, re.I) for h in heads):
        find(out, "H20", CI_FAIL, r, 1, "README has no setup/prerequisites heading",
             "validate_readme.py:116", "scan the headings")
    if not any(re.search(run, h, re.I) for h in heads):
        find(out, "H20", CI_FAIL, r, 1, "README has no run/usage heading",
             "validate_readme.py:124", "scan the headings")


def _rule_sources(root):
    """Which rule definitions actually exist in the tree under review.

    The checker must not apply a rule whose source is absent. PR #1994's head
    predates AGENTS.md and .github/policy.yml entirely, so reporting "deprecated
    model" or "frozen path" against it applies rules that did not exist when the
    branch was written. Absent source -> skip and say so.
    """
    return {
        "agents_md": os.path.exists(os.path.join(root, "AGENTS.md")),
        "policy": os.path.exists(os.path.join(root, ".github/policy.yml")),
    }


def check_layout(out, root, rel, recipe_name):
    recipe_abs = os.path.join(root, rel)
    src = _rule_sources(root)

    # H2 -- standalone ruff config anywhere in the subtree
    for dirpath, dirnames, filenames in os.walk(recipe_abs):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".venv", "node_modules")]
        for fn in filenames:
            if fn in ("ruff.toml", ".ruff.toml"):
                rp = os.path.relpath(os.path.join(dirpath, fn), root)
                find(out, "H2", CI_FAIL, rp, 1,
                     "standalone ruff config in a recipe; config lives in the repo root",
                     "python-validate-recipe.yml:268-278", "check the file exists")

    # H21 -- required files
    for f in ("README.md", "pyproject.toml", "uv.lock", ".env.example",
              "tests/test_runnability.py"):
        if not os.path.exists(os.path.join(recipe_abs, f)):
            find(out, "H21", CI_FAIL, os.path.join(rel, f), 1,
                 f"required file missing: {f}",
                 ".github/policy.yml required_files", "check the file exists")

    # H22 -- folder name
    if not re.fullmatch(r"[a-z][a-z-]*", recipe_name):
        find(out, "H22", CI_FAIL, rel, 1,
             f'folder name "{recipe_name}" must match ^[a-z][a-z-]*$',
             "validate_structure.py:84", "read the directory name")
    elif recipe_name.endswith("-"):
        find(out, "H22", CI_ADV, rel, 1,
             f'folder name "{recipe_name}" ends with a hyphen '
             "(CI permits it, the scaffolder rejects it)",
             "scaffold.py:25", "read the directory name")
    if len(recipe_name) > 30:
        find(out, "H22", CI_FAIL, rel, 1,
             f'folder name is {len(recipe_name)} chars, max is 30',
             ".github/policy.yml max_folder_name_length", "count the characters")

    # H23 -- skills/<vertical>/<solution>
    parts = rel.strip("/").split("/")
    if parts[0] == "skills" and len(parts) != 3:
        find(out, "H23", CI_FAIL, rel, 1,
             f"skills recipes must be at skills/<vertical>/<solution>/, got {rel}",
             "validate_placement.py:51-92", "count the path segments")

    # H24 -- frozen legacy roots. Only meaningful if policy.yml declares them.
    if not src["policy"]:
        SKIPPED.append(("H24", "no .github/policy.yml in the reviewed tree; "
                               "frozen paths undefined here"))
    elif re.match(r"^(python|java|go|kotlin|typescript)/agents/", rel):
        find(out, "H24", CI_FAIL, rel, 1,
             f"{rel} is under a frozen legacy path; use contrib/ or core/",
             ".github/policy.yml frozen_paths", "read the path")


def check_text_wide(out, root, rel):
    """H10 and H14 -- literal scans across the recipe."""
    recipe_abs = os.path.join(root, rel)
    if not os.path.exists(os.path.join(root, "AGENTS.md")):
        SKIPPED.append(("H10", "no AGENTS.md in the reviewed tree; the deprecated "
                               "model list is undefined here"))
        banned = None
    else:
        banned = re.compile(r"gemini-2\.0-flash|gemini-2\.5-flash")
    hits = []
    for dirpath, dirnames, filenames in (
            os.walk(recipe_abs) if banned else []):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".venv", "node_modules", ".git")]
        for fn in filenames:
            if fn in ("uv.lock", "poetry.lock"):
                continue
            fp = os.path.join(dirpath, fn)
            if os.path.getsize(fp) > 2_000_000:
                continue
            t = read(fp)
            if not t:
                continue
            for i, line in enumerate(t.split("\n"), 1):
                if banned.search(line):
                    hits.append((os.path.relpath(fp, root), i))
    if hits:
        # ONE finding, not one per hit.
        p, l = hits[0]
        find(out, "H10", CI_ADV, p, l,
             f"deprecated model id (use gemini-3.5-flash); {len(hits)} occurrence(s) "
             f"across {len({h[0] for h in hits})} file(s)",
             "AGENTS.md:40", "grep gemini-2.0-flash / gemini-2.5-flash")

    envx = os.path.join(recipe_abs, ".env.example")
    t = read(envx)
    if t:
        for i, line in enumerate(t.split("\n"), 1):
            m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
            if not m:
                continue
            var, val = m.group(1), m.group(2).strip().split("#")[0].strip()
            if var != var.upper():
                find(out, "H13", CI_FAIL, os.path.join(rel, ".env.example"), i,
                     f'"{var}" is not UPPER_SNAKE_CASE', "extract_env_vars.py:444",
                     f"read line {i}")
            if val.lower() in ("<changeme>", "changeme", "todo", "<todo>", "xxx"):
                find(out, "H14", CI_ADV, os.path.join(rel, ".env.example"), i,
                     f'placeholder should be the exact string "<TODO: update-this-value>"',
                     "extract_env_vars.py:91", f"read line {i}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--recipe", required=True, help="path relative to repo root")
    ap.add_argument("--schema", default=None,
                    help="path to manifest-schema.json (defaults to <root>/.github/schemas/)")
    ap.add_argument("--repo", help="owner/name; with --pr, fetches the changed "
                                   "files itself (paged) so nothing is truncated")
    ap.add_argument("--pr", help="PR number; use with --repo")
    ap.add_argument("--changed-files",
                    help="override: file with one PR-changed path per line. Without "
                         "this or --repo/--pr the whole recipe is audited, which is "
                         "right for a NEW recipe and wrong for a small edit.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.repo_root))
    rel = args.recipe.strip("/")
    if not os.path.isdir(os.path.join(root, rel)):
        sys.exit(f"not a directory: {os.path.join(root, rel)}")
    recipe_name = os.path.basename(rel)
    schema = args.schema or os.path.join(root, ".github/schemas/manifest-schema.json")

    global CHANGED, NEW_RECIPE
    if args.changed_files:
        CHANGED = {l.strip() for l in open(args.changed_files) if l.strip()}
    elif args.repo and args.pr:
        CHANGED = fetch_changed_files(args.repo, args.pr)
    if CHANGED is not None:
        # A PR that adds the recipe's manifest or pyproject is creating it.
        NEW_RECIPE = any(c.startswith(rel + "/") and
                         c.endswith(("manifest.yaml", "pyproject.toml"))
                         for c in CHANGED)

    pj = load_toml(os.path.join(root, rel, "pyproject.toml")) or {}
    pj_name = pj.get("project", {}).get("name", "")

    out = []
    check_pyproject(out, root, rel, recipe_name)
    check_uv_lock(out, root, rel, recipe_name, pj_name)
    check_dotenv_bootstrap(out, root, rel)
    check_manifest(out, root, rel, schema)
    check_readme(out, root, rel)
    check_layout(out, root, rel, recipe_name)
    check_text_wide(out, root, rel)
    check_env_defaults(out, root, rel)
    check_license_headers(out, root, rel)

    out.sort(key=lambda f: (f["ci"] != "fail", f["rule"], f["path"]))

    if args.json:
        print(json.dumps({"findings": out, "skipped": SKIPPED}, indent=1))
    else:
        nf = sum(1 for f in out if f["ci"] == "fail")
        print(f"{rel}: {len(out)} finding(s), {nf} CI-failing"
              if out else f"{rel}: no house-rule violations")
        if out:
            print()
        for f in out:
            tag = "FAIL" if f["ci"] == "fail" else "adv "
            print(f"  [{tag}] {f['rule']:<4} {f['path']}:{f['line']}")
            print(f"         {f['what']}")
        if FILTERED:
            from collections import Counter
            c = Counter(r for r, _ in FILTERED)
            print(f"\n  {len(FILTERED)} pre-existing violation(s) not attributed to "
                  f"this PR: {', '.join(f'{k}x{v}' for k, v in sorted(c.items()))}")
        if SKIPPED:
            print("\n  NOT CHECKED this run:")
            for rule, why in SKIPPED:
                print(f"    {rule}: {why}")
        print("\n  Never checked by this script (need an AST or judgement): "
              "H11 model literals, H12 model-var defaults, H16 noqa E402, "
              "H25 runnability assert placement")


if __name__ == "__main__":
    main()
