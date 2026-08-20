"""LLM client abstraction – backed by OpenRouter with LangSmith tracing support."""

from dataclasses import dataclass

from openai import OpenAI

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.observability.tracing import setup_tracing


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client backed by OpenRouter with LangSmith integration."""

    _OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str | None = None) -> None:
        setup_tracing()
        settings = get_settings()
        api_key = settings.openrouter_api_key
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file or environment variables."
            )
        self._model = model or settings.openrouter_model

        raw_client = OpenAI(
            api_key=api_key,
            base_url=self._OPENROUTER_BASE_URL,
        )

        # Wrap OpenAI client with LangSmith if configured
        if settings.langsmith_api_key:
            try:
                from langsmith.wrappers import wrap_openai

                self._client = wrap_openai(raw_client)
            except Exception:
                self._client = raw_client
        else:
            self._client = raw_client

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion from OpenRouter."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
