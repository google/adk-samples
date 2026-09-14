from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    """
    Configuration for the Task Planner Agent.
    """

    default_llm: str = Field(default="gemini-2.5-pro", alias="DEFAULT_LLM")

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_ignore_empty=True,
        env_file=".env",
        env_file_encoding="utf-8",
        cli_parse_args=False,
        extra="ignore",
        populate_by_name=True,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


config = AgentConfig()
