from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from agents.base import LLMClient


class OpenAICompatibleClient(LLMClient):
    """
    Minimal OpenAI-compatible chat client.

    The client is provider-agnostic. The endpoint, model and API key
    are supplied through environment variables.

    No credentials are stored in source code.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("LLM_API_KEY")

        self.base_url = (
            base_url
            or os.getenv(
                "LLM_BASE_URL",
                "https://api.openai.com/v1",
            )
        ).rstrip("/")

        self.model = model or os.getenv(
            "LLM_MODEL",
            "",
        )

        self.timeout = timeout

        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured.")

        if not self.model:
            raise ValueError("LLM_MODEL is not configured.")

    def invoke(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }

        body = json.dumps(payload).encode("utf-8")

        endpoint = f"{self.base_url}/chat/completions"

        http_request = request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": (f"Bearer {self.api_key}"),
                "Content-Type": "application/json",
            },
        )

        try:
            with request.urlopen(
                http_request,
                timeout=self.timeout,
            ) as response:
                raw = response.read().decode("utf-8")

        except error.HTTPError as exc:
            response_body = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                f"LLM API request failed: HTTP {exc.code}\n{response_body}"
            ) from exc

        except error.URLError as exc:
            raise RuntimeError(f"Could not reach LLM API: {exc.reason}") from exc

        data = json.loads(raw)

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected LLM API response format.") from exc

        if not isinstance(content, str):
            raise RuntimeError("LLM response content is not text.")

        return content
