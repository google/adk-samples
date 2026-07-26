# Terraform — `lha-skills-<env>` bucket (currently unused)

> **Status:** orphan. Workspace state now lives in Vertex Agent Runtime sandboxes
> (snapshots stored under the parent `reasoningEngine`), so this bucket is no
> longer wired into the runtime. The module is retained as a stub for future
> shared-state needs (e.g. a multi-host snapshot index).

Per-user workspace persistence is provided by `SandboxEnvironment`
(`horizon/environment/sandbox.py`): each session boots a sandbox container,
restores from the user's prior snapshot if one exists, and snapshots back at
session close. Selection is controlled by:

```bash
# Required to switch off LocalEnvironment.
export LHA_ENVIRONMENT_BACKEND=sandbox

# Required when backend=sandbox.
export AGENT_ENGINE_RESOURCE_NAME=projects/<p>/locations/<l>/reasoningEngines/<r>

# BYOC runtime image the sandbox boots; rebuilt + pushed from
# horizon/sandbox/runtime/. A new tag forces a new template.
export LHA_RUNTIME_IMAGE=us-central1-docker.pkg.dev/<p>/<repo>/runtime:<tag>

# SA email whose JWT the sandbox LB checks on every shim request.
# Caller ADC must hold roles/iam.serviceAccountTokenCreator on this SA.
export LHA_SANDBOX_CALLER_SA=lha-sandbox-caller@<p>.iam.gserviceaccount.com

# Optional: how long _close_envs_at_exit waits for snapshot/delete RPCs to
# finish before abandoning the cleanup thread (default 300s). Bump for very
# large workspaces; lower for fast shutdown.
export LHA_CLOSE_JOIN_TIMEOUT_SEC=300

# Optional: set to "1" in the CLI entrypoint so SIGTERM/SIGINT trigger the
# snapshot flush. Off by default so library/test imports don't hijack signal
# handlers from the host process.
export LHA_INSTALL_SIGNAL_HANDLERS=1
```

The snapshot index lives on the host at `~/.lha/snapshots/index.json`,
guarded by a sibling `.lock` file via `fcntl.flock` so parallel sessions for
the same `user_id` serialize their writes.

## Apply (if you still want the bucket)

```bash
cd terraform
terraform init
terraform plan -var env=dev
terraform apply -var env=dev
```

`env` ∈ `{dev, staging, prod}`. `project_id` is required — supply it with
`-var project_id=...` or `TF_VAR_project_id`.

## What this creates

- `google_storage_bucket.skills` — `lha-skills-<env>`, US multi-region,
  STANDARD class, uniform bucket-level IAM, public access blocked.
- Versioning enabled.
- Two lifecycle rules: keep ≤5 noncurrent versions; delete noncurrent
  versions older than 30 days.

## IAM (not in this module)

If you later wire the bucket back in, grant the runtime SA
`roles/storage.objectAdmin` on the bucket via the deployment terraform —
this module shouldn't know who its consumers are.

## IAP access

The Cloud Run footprint is two services:

- `lha` (backend) — `iap_enabled = false`. Invoked only by the
  `lha-web-run` SA (web frontend), the scheduler SA, and A2A peers.
  Verifies identity itself (`LHA_AUTH_MODE=iap`): the IAP JWT forwarded by
  `lha-web` as `X-LHA-IAP-Assertion`, or an end-user OAuth bearer on A2A calls
  from Gemini Enterprise.
- `lha-web` (frontend) — `iap_enabled = true`. The Express proxy in
  `web/server/` validates the IAP JWT against
  `/projects/<PROJECT_NUMBER>/locations/<REGION>/services/lha-web` and
  forwards the verified email to the backend.

Grant end-users access via the `iap_users` Terraform variable:

```bash
terraform apply -var 'iap_users=["user:you@example.com","group:eng@example.com"]'
```

For local dev, set `LHA_AUTH_MODE=dev` and `LHA_DEV_USER_ID=<email>` in
`.env` to bypass IAP and pin a stable identity.
