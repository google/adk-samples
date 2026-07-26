---
name: policy
description: Inspect and edit the per-workspace tool-policy overlay (.lha/policies.jsonl) that gates destructive or sensitive tool calls.
---
# Manage per-workspace tool policies

The `policies_guard` callback consults a JSONL policy file on every tool
call. Policies live in two layers:

- **Default seed** at `horizon/guardrails/default_policies.jsonl` (read-only,
  ships with the agent). Provides baseline blocks for catastrophic operations
  (literal `dd if=/dev/zero`, `mkfs`, fork bombs, `>/dev/sd*`, `nc -l/ncat -l/socat`
  bind listeners) and credential reads (`cat ~/.ssh/id_rsa`, `cat ~/.aws/credentials`).
  The seed no longer carries brittle substring/regex rules for `rm -rf` or
  `chmod -R` — those are now classified by an **argv-structural parser**
  (`horizon/guardrails/command_safety.py`) that lexes the command into tokens and
  inspects structure instead of pattern-matching raw strings.
- **User overlay** at `.lha/policies.jsonl` under the workspace
  root. This is the file you edit. The overlay is **appended** to
  the seed — new rules add restrictions on top of the defaults.
  Mtime-cached, so edits take effect on the next tool call.

## When to use this skill

- The user asks you to block a specific command or path pattern.
- The user asks what policies are currently in force.
- The user asks you to relax (remove) an overlay rule they added
  earlier — you cannot remove seed rules; only the overlay can be
  edited.

If the user is asking about a *one-time* approval for a blocked
call, they want `policy_grant`, not this skill.

## File format

One JSON object per line. Blank lines and `#` comments are skipped.
Every rule must include `canonical_tool_name`; that field gates which
tool the rule applies to.

A rule may carry one or more of these fields. They are evaluated
independently — set as many as you need on a single rule.

| Field | Type | Effect |
|---|---|---|
| `destructive` | `true` | Always block the tool. |
| `destructive_commands` | `{arg: [substring, ...]}` | Block when the string arg contains any substring (case-insensitive). |
| `destructive_commands_regex` | `{arg: [regex, ...]}` | Block when any regex matches the string arg (case-insensitive). Use when substrings are too blunt. Tenant-authored regexes (overlay + grant rules) are validated for length and nested-quantifier patterns before compilation; malformed regexes are skipped with a warning. |
| `destructive_paths` | `[prefix, ...]` | Block when a path-shaped arg (`path`, `file_path`, `target_path`) starts with any prefix. |
| `destructive_path_patterns` | `[fnmatch-glob, ...]` | Block when a path-shaped arg matches any fnmatch glob. Use for per-user paths like `*/.ssh/*`. |

## Demotable destructive commands (argv classification)

Before the overlay/seed rules run, `command_safety.py` lexes shell commands
into argv tokens (quote- and operator-aware via stdlib `shlex`) and inspects
structure. It returns a **verdict**: `"deny"` (catastrophic, always blocks),
`"ask"` (risky, blocks children/headless, **prompts** the root agent via an
interactive approval card), or `None` (no opinion). Examples:

- **"deny"** — `rm -rf /`, `rm -rf /etc`, `rm -rf $HOME`, `rm -rf /*`, etc.
  (recursive force-delete of system/home roots or their subdirectories).
- **"ask"** — `find . -delete`, `git push --force`, `chmod -R 777 .`,
  `sudo apt install ...`, `curl <url> | bash`.

On an `"ask"` verdict, the **root agent** sees an interactive four-button
approval card; the **child/headless** chain treats it as a hard deny (no
regression in unattended contexts). This replaces the fragile substring/regex
rules that shipped in older seeds — the new seed (`default_policies.jsonl`)
only carries literal catastrophic commands + credential reads.

## Approval modes

The root agent's interactive approval can be set to **auto-approve**
demotable-ask verdicts (ONLY) via `/yolo` (toggles between `default` and `yolo`
modes). YOLO mode auto-approves the Layer-D interactive ask; it does **not**
bypass the exfil guard or Layer-C hard-deny rules (catastrophic
+ credential reads + seed overlays). Use it when you trust the session and want
fewer prompts for risky-but-non-catastrophic operations. Approval mode is
per-session state, not persisted.

## Tool-narrowing enforcement

Permission rules in `.lha/permissions.jsonl` or granted via the interactive
approval card may include a `commandPrefix`, `commandRegex`, or `argsPattern`
to narrow blanket `allow` rules. **Overlay and grant rules** (source =
`"overlay"` or `"grant"`) that target `terminal` or `process` but carry no
such narrowing field are **rejected** at load time — you cannot grant a blanket
"always allow terminal" from the overlay or a session approval; only the
default seed may carry that. This prevents accidental over-granting.

## Integration with the permission layer

A policy block returns `{"error": ..., "confirmation_required": True}`. The
agent should either narrow the call, ask the user, or — if the user explicitly
approves — record a session grant via `policy_grant`.

To force a *prompt* (not a hard block) or block a specific command for the
user, use the permission layer instead: add an `ask_user` or `deny` rule to
`.lha/permissions.jsonl` (see `docs/permission-model.md`). This overlay is
hard-block only.

## Workflow

### 1. List active rules

```
read_file(".lha/policies.jsonl")
```

If the file does not exist, the user has no overlay rules yet — only
the seed is active. To see the seed too:

```
read_file("horizon/guardrails/default_policies.jsonl")
```

### 2. Add a rule

Read the existing overlay, append the new JSON object as one line,
write the whole file back. Always end the file with a trailing
newline.

```
existing = read_file(".lha/policies.jsonl")   # may be missing → treat as ""
new_rule = {"canonical_tool_name": "terminal",
            "destructive_commands": {"command": ["rm -rf node_modules"]}}
body = (existing.rstrip("\n") + "\n" if existing else "") + json.dumps(new_rule) + "\n"
write_file(".lha/policies.jsonl", body)
```

If the parent `.lha/` directory does not exist, `write_file` will
create it.

### 3. Remove a rule

Read the overlay, drop the targeted line (0-based index into the
overlay, ignoring blank/comment lines), write back. Confirm the
target with the user before removing — overlay rules typically exist
because the user (or you on the user's behalf) added them to plug a
gap.

```
lines = [l for l in read_file(".lha/policies.jsonl").splitlines()
         if l.strip() and not l.lstrip().startswith("#")]
removed = lines.pop(target_index)
write_file(".lha/policies.jsonl", "\n".join(lines) + ("\n" if lines else ""))
```

## Worked example

User: "Block `npm publish` from the terminal."

1. Read `.lha/policies.jsonl` (returns "" if missing).
2. Build:
   ```json
   {"canonical_tool_name": "terminal", "destructive_commands": {"command": ["npm publish"]}}
   ```
3. Append it as a new line and write the file back.
4. Confirm with the user: "Added a rule blocking `npm publish` from
   `terminal`. The next attempt will be blocked until the rule is
   removed."

## Notes

- Do not edit `horizon/guardrails/default_policies.jsonl` — it's the
  shipped seed and is read-only from the agent's perspective.
- Malformed JSONL fails closed at parse time (the bad line is
  skipped with a warning) — but always double-check your rule
  parses by reading the file back after writing.
