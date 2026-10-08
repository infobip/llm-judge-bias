"""Minimal clients for OpenAI-compatible APIs and local Hugging Face models."""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any, TypedDict

from openai import AsyncOpenAI

from .config import API_MODELS, HUGGINGFACE_MODELS


class TokenProbability(TypedDict):
    """One generated token and its top alternatives."""

    token: str
    logprob: float
    top_alternatives: list[dict[str, Any]]


class Completion(TypedDict):
    """Normalized model response."""

    content: str
    logprobs: list[TokenProbability] | None
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class ModelClient(ABC):
    """Interface shared by API and local model clients."""

    model_name: str

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        *,
        logprobs: bool = False,
    ) -> Completion:
        """Return one chat completion."""


class OpenAICompatibleClient(ModelClient):
    """OpenAI-compatible client, including LiteLLM proxies."""

    def __init__(self, model_name: str) -> None:
        endpoint = os.getenv("LITELLM_ENDPOINT", "").rstrip("/")
        api_key = os.getenv("LITELLM_API_KEY")
        if not endpoint or not api_key:
            raise RuntimeError("Set LITELLM_ENDPOINT and LITELLM_API_KEY")
        self.model_name = model_name
        self.client = AsyncOpenAI(
            base_url=endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1",
            api_key=api_key,
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        *,
        logprobs: bool = False,
    ) -> Completion:
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": min(temperature, 1.0)
            if "claude" in self.model_name.lower()
            else temperature,
            "response_format": {"type": "json_object"},
            "timeout": 60,
        }
        if logprobs:
            params.update(logprobs=True, top_logprobs=10)
        response = await self.client.chat.completions.create(**params)
        choice = response.choices[0]
        token_data = None
        if logprobs and choice.logprobs:
            token_data = [
                {
                    "token": token.token,
                    "logprob": token.logprob,
                    "top_alternatives": [
                        {"token": alt.token, "logprob": alt.logprob}
                        for alt in token.top_logprobs
                    ],
                }
                for token in choice.logprobs.content
            ]
        usage = response.usage
        return {
            "content": choice.message.content or "",
            "logprobs": token_data,
            "model": response.model,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        }


class HuggingFaceClient(ModelClient):
    """Local text-generation client using Transformers."""

    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install the 'local' dependency group") from error
        self.model_name = model_name
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        *,
        logprobs: bool = False,
    ) -> Completion:
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (KeyError, TypeError, ValueError):
            merged = [
                message.copy() for message in messages if message["role"] != "system"
            ]
            system = next(
                (
                    message["content"]
                    for message in messages
                    if message["role"] == "system"
                ),
                "",
            )
            if system and merged:
                merged[0]["content"] = f"{system}\n\n{merged[0]['content']}"
            prompt = self.tokenizer.apply_chat_template(
                merged,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]
        options: dict[str, Any] = {
            "max_new_tokens": 512,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            options.update(temperature=temperature, top_p=0.95, top_k=50)
        if logprobs:
            options.update(return_dict_in_generate=True, output_scores=True)

        def generate() -> Any:
            with self._torch.no_grad():
                return self.model.generate(**inputs, **options)

        output = await asyncio.to_thread(generate)
        sequences = output.sequences if logprobs else output
        generated = sequences[0][input_length:]
        token_data = None
        if logprobs:
            token_data = []
            for token_id, scores in zip(generated, output.scores, strict=True):
                values = self._torch.nn.functional.log_softmax(scores[0], dim=-1)
                top_values, top_ids = self._torch.topk(values, k=10)
                token_data.append(
                    {
                        "token": self.tokenizer.decode([token_id.item()]),
                        "logprob": values[token_id].item(),
                        "top_alternatives": [
                            {
                                "token": self.tokenizer.decode([item.item()]),
                                "logprob": value.item(),
                            }
                            for item, value in zip(top_ids, top_values, strict=True)
                        ],
                    }
                )
        return {
            "content": self.tokenizer.decode(
                generated, skip_special_tokens=True
            ).strip(),
            "logprobs": token_data,
            "model": self.model_name,
            "prompt_tokens": int(input_length),
            "completion_tokens": len(generated),
            "total_tokens": int(input_length + len(generated)),
        }


def get_client(model_name: str) -> ModelClient:
    """Create the configured client for a model name."""
    if model_name in API_MODELS:
        return OpenAICompatibleClient(model_name)
    if model_name in HUGGINGFACE_MODELS:
        return HuggingFaceClient(model_name)
    raise ValueError(f"Unknown model: {model_name}")
