import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class AzureConfig:
    endpoint: str
    deployment: str
    api_key: str = field(repr=False)
    reasoning_effort: str | None = None
    # OpenAI-native deployments take max_completion_tokens; OpenRouter takes max_tokens.
    token_param: str = "max_completion_tokens"

    @classmethod
    def from_env(cls) -> "AzureConfig":
        names = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_KEY")
        values = {name: os.environ.get(name, "").strip() for name in names}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError("Missing configuration: " + ", ".join(missing))
        endpoint = values[names[0]].rstrip("/")
        url = urlsplit(endpoint)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in ("", "/openai/v1")
        ):
            raise ValueError("Azure endpoint must be an HTTPS resource root or /openai/v1/ URL")
        if not url.path:
            endpoint += "/openai/v1"
        return cls(
            endpoint + "/",
            values[names[1]],
            values[names[2]],
            os.environ.get("AZURE_OPENAI_REASONING_EFFORT") or None,
        )


def openrouter_config(model: str, reasoning_effort: str | None = None) -> AzureConfig:
    """Build an OpenAI-compatible config for an OpenRouter judge (e.g. anthropic/claude-opus-5)."""
    api_key = os.environ.get("OPEN_ROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing configuration: OPEN_ROUTER_API_KEY")
    if not model.strip():
        raise ValueError("OpenRouter judge requires a model id, e.g. anthropic/claude-opus-5")
    return AzureConfig(
        endpoint="https://openrouter.ai/api/v1/",
        deployment=model.strip(),
        api_key=api_key,
        reasoning_effort=reasoning_effort,
        token_param="max_tokens",
    )


@dataclass(frozen=True)
class HydraConfig:
    database: str
    api_key: str = field(repr=False)
    base_url: str = "https://api.hydradb.com"

    @classmethod
    def from_env(cls) -> "HydraConfig":
        database = os.environ.get("HYDRA_DB_DATABASE", "").strip()
        api_key = os.environ.get("HYDRA_DB_API_KEY", "").strip()
        if not database or not api_key:
            raise ValueError("HydraDB requires HYDRA_DB_DATABASE and HYDRA_DB_API_KEY")
        endpoint = os.environ.get("HYDRA_DB_BASE_URL", "https://api.hydradb.com").rstrip("/")
        url = urlsplit(endpoint)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path
        ):
            raise ValueError("HYDRA_DB_BASE_URL must be an HTTPS API origin without a path")
        return cls(database, api_key, endpoint)
