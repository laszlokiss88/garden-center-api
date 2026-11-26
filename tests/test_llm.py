"""Focused tests for provider-native system prompts."""

import asyncio
import json

import httpx
import pytest

from garden_center.llm import (
    SYSTEM_PROMPT,
    AnthropicLLMProvider,
    LLMProviderError,
    OllamaLLMProvider,
)


def test_anthropic_uses_native_system_prompt() -> None:
    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert request.url.path == "/v1/messages"
            assert payload["system"] == SYSTEM_PROMPT
            assert payload["messages"] == [
                {
                    "role": "user",
                    "content": "Question:\nHow?\n\nSource excerpts:\nContext",
                }
            ]
            return httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "text", "text": "First part."},
                        {"type": "tool_use", "name": "ignored"},
                        {"type": "text", "text": "Second part."},
                    ]
                },
            )

        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="test-model",
            timeout_seconds=1,
            max_tokens=100,
            transport=httpx.MockTransport(handler),
        )
        try:
            assert (
                await provider.generate("How?", ["Context"])
                == "First part.\nSecond part."
            )
        finally:
            await provider.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "tool_use", "name": "ignored"}],
        [{"type": "text", "text": "   "}],
        [{"type": "text", "text": 123}],
    ],
)
def test_anthropic_rejects_responses_without_valid_text(content: object) -> None:
    async def scenario() -> None:
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="test-model",
            timeout_seconds=1,
            max_tokens=100,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"content": content})
            ),
        )
        try:
            with pytest.raises(LLMProviderError) as error:
                await provider.generate("How?", ["Context"])
            assert error.value.code == "invalid_generation_response"
        finally:
            await provider.aclose()

    asyncio.run(scenario())


def test_ollama_uses_system_role_with_chat_api() -> None:
    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert request.url.path == "/api/chat"
            assert payload["messages"] == [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Question:\nHow?\n\nSource excerpts:\nContext",
                },
            ]
            assert payload["options"]["num_predict"] == 100
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "Natural answer.",
                    }
                },
            )

        provider = OllamaLLMProvider(
            base_url="http://ollama.test",
            model="test-model",
            timeout_seconds=1,
            max_tokens=100,
            transport=httpx.MockTransport(handler),
        )
        try:
            assert await provider.generate("How?", ["Context"]) == "Natural answer."
        finally:
            await provider.aclose()

    asyncio.run(scenario())
