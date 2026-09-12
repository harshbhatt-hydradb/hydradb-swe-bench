from openai import OpenAI

from .config import AzureConfig


class AzureModel:
    def __init__(self, config: AzureConfig):
        self.config = config
        # Retry policy is bounded and recorded; tool execution is never retried by the SDK.
        self.client = OpenAI(api_key=config.api_key, base_url=config.endpoint, max_retries=2)

    def complete(
        self, messages: list[dict], tools: list[dict], *, max_tokens: int, timeout: float
    ) -> dict:
        extra = {}
        if tools:
            extra["tools"] = tools
        if self.config.reasoning_effort:
            extra["reasoning_effort"] = self.config.reasoning_effort
        result = self.client.chat.completions.create(
            model=self.config.deployment,
            messages=messages,
            max_completion_tokens=max_tokens,
            timeout=timeout,
            **extra,
        )
        return result.model_dump(exclude_none=True)

    def close(self) -> None:
        self.client.close()
