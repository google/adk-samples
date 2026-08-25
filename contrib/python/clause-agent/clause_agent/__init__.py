from dotenv import load_dotenv

# Load variables from .env if present. In production the environment is
# already populated by the platform (Cloud Run, GKE, etc.), so a missing
# .env is expected and not an error.
load_dotenv()

from clause_agent.agent import root_agent  # noqa: E402

__all__ = ["root_agent"]
