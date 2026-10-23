# Shared Vertex Agent Engine — hosts both Memory Bank (cross-session memory)
# and per-user sandboxes. One parent resource, two attached features.
#
# The google_vertex_ai_reasoning_engine terraform resource doesn't expose
# context_spec.memory_bank_config, so terraform can't create the engine
# natively. Instead we shell out to a small Python helper (idempotent: it
# discovers an existing engine by display_name or creates one) and feed the
# resulting resource name into the Cloud Run env block via
# AGENT_ENGINE_RESOURCE_NAME — both the memory and sandbox code paths read
# the same env var.
#
# The script runs on every `terraform plan` / `apply`. Vertex list is fast
# (~1 RPC) so the overhead is negligible.

data "external" "agent_engine" {
  program = [
    "uv",
    "run",
    "--directory",
    "${path.module}/..",
    "python",
    "scripts/provision_agent_engine.py",
  ]

  query = {
    project_id   = var.project_id
    location     = var.agent_engine_location
    display_name = var.agent_engine_display_name
  }
}

locals {
  agent_engine_resource_name = data.external.agent_engine.result.name
}
