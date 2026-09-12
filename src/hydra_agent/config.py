import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class AzureConfig:
    endpoint: str
    deployment: str
    api_key: str = field(repr=False)
    reasoning_effort: str | None = None

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
