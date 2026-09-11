"""House-rule checks, plus the three traps I fell into running them by hand.

The traps matter more than the happy paths: a false "this will fail CI" is the
most expensive comment this skill can produce.
"""

import json
import textwrap
from pathlib import Path

import check_house_rules as chr


def recipe(tmp_path, name="my-recipe", **files):
    root = tmp_path / "contrib" / "python" / name
    root.mkdir(parents=True)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text), encoding="utf-8")
    return str(tmp_path), f"contrib/python/{name}"


def rules(out):
    return {f["rule"] for f in out}


# ------------------------------------------------------- the three traps


def test_editable_self_reference_is_not_a_violation(tmp_path):
    """TRAP: `source = { editable = "." }` is the recipe's OWN package.

    A naive grep for `editable` flags it. I nearly filed this on PR #2373.
    """
    root, rel = recipe(
        tmp_path,
        **{
            "pyproject.toml": '[project]\nname = "my-recipe"\n',
            "uv.lock": '[[package]]\nname = "my-recipe"\nsource = { editable = "." }\n',
        },
    )
    out = []
    chr.check_uv_lock(out, root, rel, "my-recipe", "my-recipe")
    assert out == []


def test_third_party_editable_source_is_a_violation(tmp_path):
    root, rel = recipe(
        tmp_path,
        **{
            "uv.lock": '[[package]]\nname = "other-lib"\nsource = { editable = "../x" }\n'
        },
    )
    out = []
    chr.check_uv_lock(out, root, rel, "my-recipe", "my-recipe")
    assert rules(out) == {"H9"}


def test_licence_and_tags_are_permitted_manifest_keys(tmp_path):
    """TRAP: I expected these to fail the schema. They do not."""
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "properties": {
                    k: {}
                    for k in (
                        "type",
                        "status",
                        "language",
                        "description",
                        "ownership",
                        "license",
                        "tags",
                        "deployable",
                        "large",
                        "architecture",
                    )
                },
                "required": ["type"],
            }
        )
    )
    root, rel = recipe(
        tmp_path,
        **{
            "manifest.yaml": 'type: standalone\ndescription: "A real description here"\n'
            "license: Apache-2.0\ntags:\n  - finance\n"
        },
    )
    out = []
    chr.check_manifest(out, root, rel, str(schema))
    assert "H19" not in rules(out)


def test_unknown_manifest_key_is_a_violation(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"properties": {"type": {}}, "required": []}))
    root, rel = recipe(
        tmp_path, **{"manifest.yaml": "type: standalone\nowner: someone\n"}
    )
    out = []
    chr.check_manifest(out, root, rel, str(schema))
    assert "H19" in rules(out)


def test_load_dotenv_reports_where_it_is_called_not_just_absence(tmp_path):
    """TRAP: I posted "only in tests/ and eval/" from a truncated grep. Wrong.

    It was called from four package modules. The finding must name them, or the
    author replies "it's right there in agent.py" and discounts the comment.
    """
    root, rel = recipe(
        tmp_path,
        **{
            "mypkg/__init__.py": "# no bootstrap here\n",
            "mypkg/agent.py": "from dotenv import load_dotenv\nload_dotenv()\n",
            "mypkg/tools.py": "from dotenv import load_dotenv\nload_dotenv()\n",
        },
    )
    out = []
    chr.check_dotenv_bootstrap(out, root, rel)
    assert rules(out) == {"H15"}
    what = out[0]["what"]
    assert "2 module(s)" in what
    assert "agent.py" in what and "tools.py" in what


def test_load_dotenv_in_package_init_is_compliant(tmp_path):
    root, rel = recipe(
        tmp_path,
        **{
            "mypkg/__init__.py": "from dotenv import load_dotenv\nload_dotenv()\n",
            "mypkg/agent.py": "import os\n",
        },
    )
    out = []
    chr.check_dotenv_bootstrap(out, root, rel)
    assert out == []


def test_recipe_that_never_reads_dotenv_is_not_flagged(tmp_path):
    root, rel = recipe(
        tmp_path, **{"mypkg/__init__.py": "", "mypkg/a.py": "x = 1\n"}
    )
    out = []
    chr.check_dotenv_bootstrap(out, root, rel)
    assert out == []


# -------------------------------------------------------------- pyproject


def test_ruff_table_in_recipe_is_ci_fail(tmp_path):
    root, rel = recipe(
        tmp_path,
        **{
            "pyproject.toml": '[project]\nname = "my-recipe"\nrequires-python = ">=3.11"\n[tool.ruff]\nline-length = 100\n'
        },
    )
    out = []
    chr.check_pyproject(out, root, rel, "my-recipe")
    assert "H1" in rules(out)
    assert next(f for f in out if f["rule"] == "H1")["ci"] == "fail"


def test_name_must_equal_folder_basename(tmp_path):
    root, rel = recipe(
        tmp_path, **{"pyproject.toml": '[project]\nname = "other"\n'}
    )
    out = []
    chr.check_pyproject(out, root, rel, "my-recipe")
    assert "H3" in rules(out)


def test_requires_python_310_permits_older(tmp_path):
    root, rel = recipe(
        tmp_path,
        **{
            "pyproject.toml": '[project]\nname = "my-recipe"\nrequires-python = ">=3.10"\n'
        },
    )
    out = []
    chr.check_pyproject(out, root, rel, "my-recipe")
    assert "H4" in rules(out)


def test_requires_python_312_excludes_311(tmp_path):
    root, rel = recipe(
        tmp_path,
        **{
            "pyproject.toml": '[project]\nname = "my-recipe"\nrequires-python = ">=3.12"\n'
        },
    )
    out = []
    chr.check_pyproject(out, root, rel, "my-recipe")
    assert "H4" in rules(out)


def test_requires_python_311_range_is_accepted(tmp_path):
    root, rel = recipe(
        tmp_path,
        **{
            "pyproject.toml": '[project]\nname = "my-recipe"\nrequires-python = ">=3.11,<3.14"\n'
        },
    )
    out = []
    chr.check_pyproject(out, root, rel, "my-recipe")
    assert "H4" not in rules(out)


def test_dotenv_in_dev_group_does_not_count(tmp_path):
    root, rel = recipe(
        tmp_path,
        **{
            "pyproject.toml": '[project]\nname = "my-recipe"\nrequires-python = ">=3.11"\n'
            'dependencies = ["google-adk>=2.0"]\n'
            '[dependency-groups]\ndev = ["python-dotenv>=1.0.0"]\n'
        },
    )
    out = []
    chr.check_pyproject(out, root, rel, "my-recipe")
    assert "H6" in rules(out)


def test_single_bracket_uv_index_is_flagged(tmp_path):
    root, rel = recipe(
        tmp_path,
        **{
            "pyproject.toml": '[project]\nname = "my-recipe"\nrequires-python = ">=3.11"\n'
            '[tool.uv.index]\nurl = "https://pypi.org/simple"\ndefault = true\n'
        },
    )
    out = []
    chr.check_pyproject(out, root, rel, "my-recipe")
    assert "H5" in rules(out)


# ------------------------------------------------------------ model ids


def test_deprecated_model_id_is_one_finding_not_one_per_hit(tmp_path):
    root, rel = recipe(
        tmp_path,
        **{
            "a.py": 'm = "gemini-2.5-flash"\n',
            "b.py": 'n = "gemini-2.5-flash"\n',
            "c.md": "use gemini-2.0-flash\n",
        },
    )
    # H10 only applies where AGENTS.md declares the deprecated list.
    (Path(root) / "AGENTS.md").write_text("Do NOT use gemini-2.5-flash.\n")
    out = []
    chr.check_text_wide(out, root, rel)
    h10 = [f for f in out if f["rule"] == "H10"]
    assert len(h10) == 1
    assert "3 occurrence(s)" in h10[0]["what"]


def test_lockfile_is_not_scanned_for_model_ids(tmp_path):
    root, rel = recipe(tmp_path, **{"uv.lock": "gemini-2.5-flash\n"})
    (Path(root) / "AGENTS.md").write_text("Do NOT use gemini-2.5-flash.\n")
    out = []
    chr.check_text_wide(out, root, rel)
    assert "H10" not in rules(out)


# ------------------------------------------------------- skip reporting


def test_skipped_rules_are_reported_never_silent(tmp_path):
    """A checker that hides what it did not check makes every clean run a lie."""
    chr.SKIPPED.clear()
    root, rel = recipe(tmp_path, **{"manifest.yaml": "type: standalone\n"})
    chr.check_manifest([], root, rel, "/nonexistent/schema.json")
    # pyyaml absent -> degraded parse must be recorded, not swallowed
    import importlib.util

    if importlib.util.find_spec("yaml") is None:
        assert any("H19" in r for r, _ in chr.SKIPPED)


# --------------------------------- change-awareness and rule-source presence


def test_preexisting_violation_is_not_attributed_to_a_small_edit(tmp_path):
    """PR #1994 exposed this: a 16-line model-name change was being blamed for
    [tool.ruff] tables and missing test files it never touched."""
    root, rel = recipe(
        tmp_path,
        **{
            "pyproject.toml": '[project]\nname = "my-recipe"\nrequires-python = ">=3.10"\n[tool.ruff]\nx = 1\n'
        },
    )
    changed = tmp_path / "changed.txt"
    changed.write_text(f"{rel}/some_other_file.py\n")
    chr.CHANGED = {f"{rel}/some_other_file.py"}
    chr.NEW_RECIPE = False
    chr.FILTERED.clear()
    try:
        out = []
        chr.check_pyproject(out, root, rel, "my-recipe")
        assert out == [], (
            "pre-existing pyproject violations must not be reported"
        )
        assert chr.FILTERED, "and they must be counted, not silently dropped"
    finally:
        chr.CHANGED = None
        chr.NEW_RECIPE = True


def test_new_recipe_is_audited_in_full(tmp_path):
    root, rel = recipe(
        tmp_path,
        **{
            "pyproject.toml": '[project]\nname = "wrong"\nrequires-python = ">=3.10"\n'
        },
    )
    chr.CHANGED = {f"{rel}/pyproject.toml"}
    chr.NEW_RECIPE = True
    try:
        out = []
        chr.check_pyproject(out, root, rel, "my-recipe")
        assert {f["rule"] for f in out} >= {"H3", "H4"}
    finally:
        chr.CHANGED = None


def test_h10_is_skipped_when_agents_md_is_absent(tmp_path):
    """Applying today's deprecated-model list to a branch that predates it is
    anachronistic. PR #1994's head has no AGENTS.md at all."""
    root, rel = recipe(tmp_path, **{"a.py": 'm = "gemini-2.5-flash"\n'})
    chr.SKIPPED.clear()
    out = []
    chr.check_text_wide(out, root, rel)
    assert "H10" not in rules(out)
    assert any(r == "H10" for r, _ in chr.SKIPPED)


def test_h10_fires_when_agents_md_declares_the_list(tmp_path):
    root, rel = recipe(tmp_path, **{"a.py": 'm = "gemini-2.5-flash"\n'})
    (Path(root) / "AGENTS.md").write_text("Do NOT use gemini-2.5-flash.\n")
    chr.SKIPPED.clear()
    out = []
    chr.check_text_wide(out, root, rel)
    assert "H10" in rules(out)


def test_h24_is_skipped_without_policy_yml(tmp_path):
    root, _rel = recipe(tmp_path, name="legacy")
    frozen_rel = "python/agents/legacy"
    (Path(root) / "python" / "agents" / "legacy").mkdir(parents=True)
    chr.SKIPPED.clear()
    out = []
    chr.check_layout(out, root, frozen_rel, "legacy")
    assert "H24" not in rules(out)
    assert any(r == "H24" for r, _ in chr.SKIPPED)


def test_new_recipe_detection_needs_the_complete_file_list(tmp_path):
    """PR #2302 regression: `gh pr view --json files` caps at 100.

    With a truncated list, pyproject.toml was absent, NEW_RECIPE came out False,
    and every finding on a brand-new 787-file recipe was filtered as pre-existing.
    """
    root, rel = recipe(
        tmp_path,
        **{
            "pyproject.toml": '[project]\nname = "my-recipe"\nrequires-python = ">=3.10"\n'
        },
    )

    truncated = {f"{rel}/some/other/file{i}.py" for i in range(100)}
    chr.CHANGED, chr.NEW_RECIPE = (
        truncated,
        any(
            c.startswith(rel + "/")
            and c.endswith(("manifest.yaml", "pyproject.toml"))
            for c in truncated
        ),
    )
    chr.FILTERED.clear()
    try:
        out = []
        chr.check_pyproject(out, root, rel, "my-recipe")
        assert out == [], (
            "truncated list should look like an edit, not a new recipe"
        )
    finally:
        chr.CHANGED, chr.NEW_RECIPE = None, True

    complete = truncated | {f"{rel}/pyproject.toml"}
    chr.CHANGED, chr.NEW_RECIPE = complete, True
    try:
        out = []
        chr.check_pyproject(out, root, rel, "my-recipe")
        assert "H4" in rules(out), (
            "complete list must attribute findings to the PR"
        )
    finally:
        chr.CHANGED, chr.NEW_RECIPE = None, True


# ------------------------------------------------ H26: env-read defaults


def env_hits(src):
    return chr._env_read_defaults(src, "a.py")


def test_getenv_with_a_default_is_flagged():
    assert env_hits('import os\nx = os.getenv("X", "d")\n')


def test_environ_get_with_a_default_is_flagged():
    assert env_hits('import os\nx = os.environ.get("X", "d")\n')


def test_setdefault_is_flagged():
    assert env_hits('import os\nos.environ.setdefault("X", "d")\n')


def test_keyword_default_is_flagged():
    assert env_hits('import os\nx = os.getenv("X", default="d")\n')


def test_or_fallback_is_flagged():
    assert env_hits('import os\nx = os.getenv("X") or "fallback"\n')


def test_bare_getenv_is_clean():
    assert not env_hits('import os\nx = os.getenv("X")\n')


def test_dict_style_access_is_clean():
    """os.environ["X"] cannot carry a default, so it is the encouraged shape."""
    assert not env_hits('import os\nx = os.environ["X"]\n')


def test_or_none_is_not_a_default():
    assert not env_hits('import os\nx = os.environ.get("X") or None\n')


def test_a_default_in_a_docstring_is_not_a_finding():
    """AST, not regex -- prose describing the pattern must not fire."""
    assert not env_hits(
        '"""Call os.getenv("X", "d") to read it."""\nimport os\n'
    )


def test_an_unrelated_dot_get_is_not_an_env_read():
    assert not env_hits('d = {}\nx = d.get("X", "default")\n')


def test_syntax_error_does_not_crash_the_check():
    assert env_hits("def broken(:\n") == []


# ------------------------------------------------ H27: licence headers

FULL = "\n".join(
    "# " + ln if ln else "#"
    for ln in [
        "Copyright 2026 Google LLC",
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
)


def test_full_header_is_recognised():
    assert chr._header_state(FULL + "\nimport os\n", "#")[0] == "full"


def test_one_line_copyright_is_partial():
    assert (
        chr._header_state("# Copyright 2026 Google LLC\nimport os\n", "#")[0]
        == "partial"
    )


def test_no_header_is_none():
    assert chr._header_state("import os\n", "#")[0] == "none"


def test_header_after_a_ruff_pragma_still_counts():
    assert chr._header_state("# ruff: noqa\n" + FULL, "#")[0] == "full"


def test_slash_comment_style_is_supported():
    js = FULL.replace("#", "//")
    assert chr._header_state(js, "//")[0] == "full"


def test_a_type_with_no_headers_anywhere_is_not_drift(tmp_path):
    """Every .tf file lacking a header is the convention, not 12 violations.

    Pooling extensions produced exactly this false positive on PR #2302.
    """
    root, rel = recipe(
        tmp_path,
        **{f"terraform/f{i}.tf": 'resource "x" {}\n' for i in range(6)},
    )
    out = []
    chr.check_license_headers(out, root, rel)
    assert out == []


def test_a_minority_breaking_the_convention_is_flagged(tmp_path):
    files = {f"m{i}.py": FULL + "\nimport os\n" for i in range(8)}
    files["odd.py"] = "# Copyright 2026 Google LLC\nimport os\n"
    root, rel = recipe(tmp_path, **files)
    out = []
    chr.check_license_headers(out, root, rel)
    assert "H27" in rules(out)
    assert "1 .py file(s) differ" in out[0]["what"]


def test_minority_breaking_a_clear_convention_says_so(tmp_path):
    files = {f"m{i}.py": FULL + "\nimport os\n" for i in range(8)}
    files["odd.py"] = "# Copyright 2026 Google LLC\nimport os\n"
    root, rel = recipe(tmp_path, **files)
    out = []
    chr.check_license_headers(out, root, rel)
    assert "differ from the 8" in out[0]["what"]


def test_two_competing_conventions_are_described_as_such(tmp_path):
    """PR #2373: 78 files use a one-line notice, 9 use Apache. Calling the 78
    'truncated Apache headers' misdescribes what is actually there."""
    files = {
        f"short{i}.py": "# Copyright 2026 Google LLC\nimport os\n"
        for i in range(8)
    }
    files.update({f"full{i}.py": FULL + "\nimport os\n" for i in range(2)})
    root, rel = recipe(tmp_path, **files)
    out = []
    chr.check_license_headers(out, root, rel)
    assert "two different headers" in out[0]["what"]


def test_no_apache_header_anywhere_is_reported_once(tmp_path):
    files = {f"m{i}.py": "import os\n" for i in range(6)}
    root, rel = recipe(tmp_path, **files)
    out = []
    chr.check_license_headers(out, root, rel)
    assert len(out) == 1
    assert "no .py file in this recipe" in out[0]["what"]


def test_config_formats_stay_consistency_only(tmp_path):
    """Required for source, consistency-only for config: 6 header-less .tf files
    are the terraform convention, not six violations."""
    root, rel = recipe(
        tmp_path,
        **{f"terraform/f{i}.tf": 'resource "x" {}\n' for i in range(6)},
    )
    out = []
    chr.check_license_headers(out, root, rel)
    assert out == []


def test_a_single_source_file_establishes_nothing(tmp_path):
    root, rel = recipe(tmp_path, **{"only.py": "import os\n"})
    out = []
    chr.check_license_headers(out, root, rel)
    assert out == []


# ------------------------------------------------------------------ H48


def manifest_with(tmp_path, ownership, name="my-recipe"):
    return recipe(
        tmp_path,
        name=name,
        **{
            "manifest.yaml": "type: standalone\n"
            'description: "A real description here"\n'
            f"ownership:\n{ownership}"
        },
    )


def h48(out):
    return [f for f in out if f["rule"] == "H48"]


def test_company_name_as_team_is_reported(tmp_path):
    for value in (
        '"google"',
        '"Google LLC"',
        '"Google Cloud"',
        '"ADK Samples Team"',
        '"n/a"',
        '"me"',
        '"TEAM"',
    ):
        out = []
        root, rel = manifest_with(
            tmp_path / value.strip('"'), f'  team: {value}\n  poc: "someone"\n'
        )
        chr.check_manifest(out, root, rel, None)
        assert h48(out), f"{value} should be flagged as a non-team"


def test_team_equal_to_the_poc_handle_is_reported(tmp_path):
    out = []
    root, rel = manifest_with(
        tmp_path, '  team: "lspataroG"\n  poc: "lspatarog"\n'
    )
    chr.check_manifest(out, root, rel, None)
    assert "poc" in h48(out)[0]["what"]
    assert "cannot be both" in h48(out)[0]["what"]


def test_team_equal_to_a_contributor_handle_is_reported(tmp_path):
    out = []
    root, rel = manifest_with(
        tmp_path,
        '  team: "someone"\n  poc: "other"\n  contributors:\n    - "someone"\n',
    )
    chr.check_manifest(out, root, rel, None)
    assert h48(out), "a team matching a contributor handle should be flagged"


def test_real_team_names_already_in_the_repo_are_not_flagged(tmp_path):
    """Calibration. A 'that looks like a username' heuristic flags every one of
    these, and each is a real owning team in google/adk-samples today."""
    for value in (
        '"DEE"',
        '"octo"',
        '"adk-kotlin"',
        '"attenu-io"',
        '"OpenEAGO"',
        '"RobustAI"',
        '"Verizon"',
        "FDE/Blackbelt",
        '"GS&I AI Apps and Platforms"',
        '"Developer Evangelism & Engineering (DevRel) - Cloud AI"',
    ):
        out = []
        root, rel = manifest_with(
            tmp_path / value.strip('"').replace("/", "-"),
            f'  team: {value}\n  poc: "someone"\n',
        )
        chr.check_manifest(out, root, rel, None)
        assert not h48(out), f"{value} is a real team and must not be flagged"


def test_placeholder_team_stays_h17_only(tmp_path):
    """Two comments on one line is noise; the placeholder is H17's finding."""
    out = []
    root, rel = manifest_with(
        tmp_path,
        '  team: "TODO: Replace with your team name"\n  poc: "someone"\n',
    )
    chr.check_manifest(out, root, rel, None)
    assert "H17" in rules(out) and not h48(out)


def test_h48_survives_the_changed_file_filter(tmp_path, monkeypatch):
    """It is reported even when the PR did not touch manifest.yaml -- the whole
    point of the rule is that nothing else ever catches the value."""
    monkeypatch.setattr(chr, "CHANGED", {"contrib/python/my-recipe/agent.py"})
    monkeypatch.setattr(chr, "NEW_RECIPE", False)
    out = []
    root, rel = manifest_with(tmp_path, '  team: "google"\n  poc: "someone"\n')
    chr.check_manifest(out, root, rel, None)
    assert h48(out)


# ---------------------------------------------- the degraded (no pyyaml) parse

TEMPLATE_MANIFEST = """\
type: "standalone"     # Options: [standalone | module]
status: "active"        # Options: [active | inactive]
language: "python"      # Options: [python | java | go | kotlin | typescript]
description: "A real description of a real recipe, long enough to pass."
# deployable: true      # (optional) omit if false
ownership:
  team: "google"
  poc: "someone"
"""

FULL_SCHEMA = {
    "properties": {
        "type": {"enum": ["standalone", "module"]},
        "status": {"enum": ["active", "inactive"]},
        "language": {"enum": ["python", "java", "go", "kotlin", "typescript"]},
        "description": {},
        "ownership": {},
        "deployable": {},
        "license": {},
        "tags": {},
        "architecture": {},
        "large": {},
    },
    "required": ["type", "status", "language", "description", "ownership"],
}


def test_inline_comments_do_not_fake_an_enum_violation(tmp_path, monkeypatch):
    """The manifest TEMPLATE ships every enum with a trailing `# Options: [...]`
    comment. A fallback parser that keeps it reads the value as
    '"standalone"  # Options: [standalone | module]' and H19 reports three
    CI-FAILs that are pure fiction. It did, on contrib/python/clause-agent."""
    monkeypatch.setitem(
        __import__("sys").modules, "yaml", None
    )  # force degraded
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(FULL_SCHEMA))
    root, rel = recipe(tmp_path, **{"manifest.yaml": TEMPLATE_MANIFEST})
    out = []
    chr.check_manifest(out, root, rel, str(schema))
    assert "H19" not in rules(out), [f["what"] for f in out]


def test_degraded_parse_still_reads_the_ownership_block(tmp_path, monkeypatch):
    """Degrading must not silently disable H48 -- a checker that quietly skips
    a rule is worse than no checker."""
    monkeypatch.setitem(__import__("sys").modules, "yaml", None)
    root, rel = recipe(tmp_path, **{"manifest.yaml": TEMPLATE_MANIFEST})
    out = []
    chr.check_manifest(out, root, rel, None)
    assert h48(out) and 'is "google"' in h48(out)[0]["what"]


def test_a_commented_out_key_is_not_a_key(tmp_path):
    """`# deployable: true` in the template is documentation, not a declaration."""
    assert (
        chr._parse_manifest("# deployable: true\ntype: standalone\n")[0].get(
            "deployable"
        )
        is None
    )


def test_scalar_values_survive_a_hash_inside_them():
    """A `#` that is not preceded by whitespace is part of the value."""
    assert chr._scalar_value("issue-#42  # a real comment") == "issue-#42"
    assert chr._scalar_value('"quoted # hash"  # comment') == "quoted # hash"
    assert chr._scalar_value("plain value") == "plain value"


# --------------------------------- H39 H40 H41 H42 H43 H44 H47 (the silent gap)


def test_h39_a_stub_committed_as_a_real_value(tmp_path):
    root, rel = recipe(
        tmp_path,
        **{
            ".env.example": "GOOGLE_CLOUD_PROJECT=your-project-id\nREAL_ONE=us-central1\n"
        },
    )
    out = []
    chr.check_text_wide(out, root, rel)
    hits = [f for f in out if f["rule"] == "H39"]
    assert len(hits) == 1 and "your-project-id" in hits[0]["what"]


def test_h39_does_not_fire_on_a_stub_shaped_substring(tmp_path):
    """A URL that CONTAINS example.com is a real value; the stub is the whole
    value. A substring rule flags every docs link in every recipe."""
    root, rel = recipe(
        tmp_path,
        **{
            ".env.example": "DOCS_URL=https://example.com/docs/getting-started\n"
            "REGION=us-central1\nTIMEOUT=30\n"
        },
    )
    out = []
    chr.check_text_wide(out, root, rel)
    assert not [f for f in out if f["rule"] == "H39"]


def test_h39_and_h14_never_both_fire_on_one_line(tmp_path):
    root, rel = recipe(tmp_path, **{".env.example": "KEY=changeme\n"})
    out = []
    chr.check_text_wide(out, root, rel)
    assert [f["rule"] for f in out] == ["H14"]


def test_h40_language_disagrees_with_the_path(tmp_path):
    root = tmp_path
    (root / "core" / "python" / "thing").mkdir(parents=True)
    (root / "core" / "python" / "thing" / "manifest.yaml").write_text(
        'type: standalone\nlanguage: "go"\n'
    )
    out = []
    chr.check_layout(out, str(root), "core/python/thing", "thing")
    hits = [f for f in out if f["rule"] == "H40"]
    assert len(hits) == 1 and '"go"' in hits[0]["what"]


def test_h40_stays_quiet_when_the_path_has_no_language_segment(tmp_path):
    """core/<name>/ is a placement problem, not a language mismatch. Calling it
    one describes the wrong defect to the author."""
    root = tmp_path
    (root / "core" / "thing").mkdir(parents=True)
    (root / "core" / "thing" / "manifest.yaml").write_text(
        'type: standalone\nlanguage: "python"\n'
    )
    out = []
    chr.check_layout(out, str(root), "core/thing", "thing")
    assert not [f for f in out if f["rule"] == "H40"]


def test_h41_and_h23_and_h47_never_double_report(tmp_path):
    """One misplaced skill, one comment -- not three saying the same thing."""
    cases = {
        "skills/store-ops": "H41",  # too shallow
        "skills/a/b/c": "H23",  # too deep
        "skills/python/store-ops": "H47",  # right depth, language folder
        "skills/retail/store-ops": None,  # correct
    }
    for rel, expected in cases.items():
        root = tmp_path / rel.replace("/", "_")
        (root / rel).mkdir(parents=True)
        out = []
        chr.check_layout(out, str(root), rel, rel.rsplit("/", 1)[-1])
        got = [f["rule"] for f in out if f["rule"] in ("H23", "H41", "H47")]
        assert got == ([expected] if expected else []), f"{rel} -> {got}"


def test_h42_mixed_pr_is_reported_once_anchored_in_the_recipe(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        chr,
        "CHANGED",
        {
            ".agents/skills/some-skill/SKILL.md",
            "contrib/python/my-recipe/manifest.yaml",
            "contrib/python/my-recipe/agent.py",
        },
    )
    out = []
    root, rel = recipe(tmp_path)
    chr.check_pr_shape(out, root, rel)
    assert len(out) == 1
    assert out[0]["rule"] == "H42"
    assert out[0]["path"] == "contrib/python/my-recipe/manifest.yaml"


def test_h42_silent_on_a_skill_only_or_recipe_only_pr(tmp_path, monkeypatch):
    root, rel = recipe(tmp_path)
    for changed in (
        {".agents/skills/x/SKILL.md"},
        {"contrib/python/my-recipe/agent.py"},
        None,
    ):
        monkeypatch.setattr(chr, "CHANGED", changed)
        out = []
        chr.check_pr_shape(out, root, rel)
        assert out == [], changed


def test_h43_only_reports_files_git_actually_tracks(tmp_path, monkeypatch):
    """An untracked .env is a developer who ran the recipe. Reporting it as
    committed is the most alarming false positive this script could produce --
    on the real repo the filesystem shows 17 and git tracks none."""
    root, rel = recipe(
        tmp_path, **{".env": "SECRET=x\n", ".env.example": "A=b\n"}
    )
    monkeypatch.setattr(chr, "_TRACKED", {})
    monkeypatch.setattr(chr, "_tracked_files", lambda r, s: set())
    out = []
    chr.check_layout(out, root, rel, "my-recipe")
    assert not [f for f in out if f["rule"] == "H43"]

    monkeypatch.setattr(
        chr,
        "_tracked_files",
        lambda r, s: {f"{rel}/.env", f"{rel}/.env.example"},
    )
    out = []
    chr.check_layout(out, root, rel, "my-recipe")
    hits = [f for f in out if f["rule"] == "H43"]
    assert len(hits) == 1 and hits[0]["path"].endswith("/.env")


def test_h43_example_variants_are_not_real_env_files(tmp_path, monkeypatch):
    root, rel = recipe(tmp_path)
    monkeypatch.setattr(chr, "_TRACKED", {})
    monkeypatch.setattr(
        chr,
        "_tracked_files",
        lambda r, s: {
            f"{rel}/.env.local.example",
            f"{rel}/.env.test.sample",
            f"{rel}/.env.template",
        },
    )
    out = []
    chr.check_layout(out, root, rel, "my-recipe")
    assert not [f for f in out if f["rule"] == "H43"]


def test_h43_says_so_when_git_cannot_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(chr, "SKIPPED", [])
    monkeypatch.setattr(chr, "_tracked_files", lambda r, s: None)
    root, rel = recipe(tmp_path)
    chr.check_layout([], root, rel, "my-recipe")
    assert any(r == "H43" for r, _ in chr.SKIPPED)


def test_h44_pruned_directory_name(tmp_path, monkeypatch):
    """From git, not the filesystem: dist/ and build/ are exactly what a local
    build leaves behind and .gitignore hides, and SKILL.md invokes this
    checker against a developer's own tree."""
    root, rel = recipe(tmp_path, **{"dist/thing.py": "x = 1\n"})
    monkeypatch.setattr(chr, "_TRACKED", {})
    monkeypatch.setattr(
        chr, "_tracked_files", lambda r, s: {f"{rel}/dist/thing.py"}
    )
    out = []
    chr.check_layout(out, root, rel, "my-recipe")
    hits = [f for f in out if f["rule"] == "H44"]
    assert len(hits) == 1 and hits[0]["path"].endswith("/dist")


def test_h44_ignores_an_untracked_build_directory(tmp_path, monkeypatch):
    root, rel = recipe(tmp_path, **{"dist/thing.py": "x = 1\n"})
    monkeypatch.setattr(chr, "_TRACKED", {})
    monkeypatch.setattr(chr, "_tracked_files", lambda r, s: set())
    out = []
    chr.check_layout(out, root, rel, "my-recipe")
    assert not [f for f in out if f["rule"] == "H44"]


def test_h43_reports_one_grouped_finding_not_one_per_file(
    tmp_path, monkeypatch
):
    """H10 and H26 both group; a recipe with twelve stray files should not
    collect twelve comments."""
    root, rel = recipe(tmp_path)
    monkeypatch.setattr(chr, "_TRACKED", {})
    monkeypatch.setattr(
        chr,
        "_tracked_files",
        lambda r, s: {f"{rel}/.env", f"{rel}/.DS_Store", f"{rel}/a.pyc"},
    )
    out = []
    chr.check_layout(out, root, rel, "my-recipe")
    hits = [f for f in out if f["rule"] == "H43"]
    assert len(hits) == 1
    assert "3 such file(s)" in hits[0]["what"]


def test_h43_does_not_call_every_pem_a_private_key(tmp_path, monkeypatch):
    """A public CA bundle is an ordinary committed file, and "you committed a
    private key" is the most alarming thing this checker can say."""
    root, rel = recipe(tmp_path)
    monkeypatch.setattr(chr, "_TRACKED", {})
    monkeypatch.setattr(
        chr,
        "_tracked_files",
        lambda r, s: {
            f"{rel}/certs/server-ca.pem",
            f"{rel}/service-account-template.json",
            f"{rel}/.vscode/launch.json",
        },
    )
    out = []
    chr.check_layout(out, root, rel, "my-recipe")
    assert not [f for f in out if f["rule"] == "H43"]


def test_h43_still_catches_a_real_private_key(tmp_path, monkeypatch):
    root, rel = recipe(tmp_path)
    monkeypatch.setattr(chr, "_TRACKED", {})
    monkeypatch.setattr(
        chr, "_tracked_files", lambda r, s: {f"{rel}/server-key.pem"}
    )
    out = []
    chr.check_layout(out, root, rel, "my-recipe")
    assert [f for f in out if f["rule"] == "H43"]


def test_tracked_file_cache_is_per_recipe(tmp_path):
    """The CI lane checks several recipes in one process. A single cached set
    answers the second recipe with the first recipe's files."""
    chr._TRACKED.clear()
    chr._TRACKED[("/root", "a")] = {"a/.env"}
    chr._TRACKED[("/root", "b")] = set()
    assert chr._tracked_files("/root", "b") == set()
    assert chr._tracked_files("/root", "a") == {"a/.env"}


def test_words_that_name_real_teams_are_not_generic(tmp_path):
    """The trailing-noun stripper is why this list must stay short: "Cloud
    Org" strips to "cloud", so a plausible team word on the list flags every
    real team whose name ends in it. `DevRel` is the everyday short form of a
    team that owns five recipes in this repository."""
    for value in (
        "DevRel",
        "Community",
        "Engineering",
        "Cloud Org",
        "Samples",
        "Developer Relations",
        "Public Sector",
    ):
        out = []
        root, rel = manifest_with(
            tmp_path / value.replace(" ", "-"),
            f'  team: "{value}"\n  poc: "someone"\n',
        )
        chr.check_manifest(out, root, rel, None)
        assert not h48(out), f"{value} is a plausible real team"


def test_a_group_alias_is_an_owner(tmp_path):
    """A mailing list is the one value immune to the failure this rule exists
    to prevent: it does not leave when a person does."""
    out = []
    root, rel = manifest_with(
        tmp_path, '  team: "adk-samples-team@google.com"\n  poc: "someone"\n'
    )
    chr.check_manifest(out, root, rel, None)
    assert not h48(out)


def test_a_url_is_still_not_a_team(tmp_path):
    out = []
    root, rel = manifest_with(
        tmp_path, '  team: "https://github.com/orgs/x/teams/y"\n  poc: "a"\n'
    )
    chr.check_manifest(out, root, rel, None)
    assert h48(out)
