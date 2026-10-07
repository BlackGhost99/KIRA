"""Fournisseur Anthropic (API Messages), avec outils et mise en cache du prompt système."""
from __future__ import annotations

from .. import config, net
from .errors import http_error
from .base import LLMError, LLMResult, Provider, ToolCall, ToolSpec

URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


def convert_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    pending: list[dict] = []

    def flush() -> None:
        if pending:
            out.append({"role": "user", "content": list(pending)})
            pending.clear()

    for m in messages:
        role = m["role"]
        if role == "tool":
            content: object = m.get("content") or "(vide)"
            if m.get("images"):
                content = [{"type": "text", "text": content}] + [
                    {"type": "image", "source": {"type": "base64", "media_type": img["mime"], "data": img["b64"]}}
                    for img in m["images"]
                ]
            pending.append({"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": content})
            continue
        flush()
        if role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
            if blocks:
                out.append({"role": "assistant", "content": blocks})
    flush()
    return out


class AnthropicProvider(Provider):
    name = "anthropic"

    def fingerprint(self) -> str:
        import hashlib
        s = config.settings
        return hashlib.sha256(repr((s.anthropic_api_key, s.anthropic_model, s.anthropic_model_fast, s.anthropic_model_deep)).encode()).hexdigest()

    def configured(self) -> bool:
        return bool(config.settings.anthropic_api_key)

    def model_for(self, tier: str) -> str:
        s = config.settings
        return {"fast": s.anthropic_model_fast, "deep": s.anthropic_model_deep}.get(tier, s.anthropic_model)

    def complete(self, system, messages, tools, tier, max_tokens) -> LLMResult:
        model = self.model_for(tier)
        blocks = [{"type": "text", "text": t} for t in system if t and t.strip()]
        if blocks:
            blocks[0]["cache_control"] = {"type": "ephemeral"}  # la partie stable du prompt est mise en cache
        body: dict = {"model": model, "max_tokens": max_tokens, "messages": convert_messages(messages)}
        if blocks:
            body["system"] = blocks
        if tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]
        resp = net.session().post(
            URL,
            headers={
                "x-api-key": config.settings.anthropic_api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
            json=body,
            allow_redirects=False,
            timeout=(10, max(5, min(120, config.settings.llm_timeout))),
        )
        if resp.status_code != 200:
            raise http_error(resp, self.name)
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError("anthropic : JSON invalide", provider=self.name) from exc
        try:
            return self._parse_response(data, model)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise LLMError("anthropic : réponse invalide", provider=self.name) from exc

    def _parse_response(self, data, model) -> LLMResult:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(block["id"], block["name"], block.get("input") or {}))
        if not text_parts and not calls:
            raise LLMError("anthropic : réponse vide", provider=self.name)
        usage = data.get("usage", {})
        return LLMResult(
            text="".join(text_parts).strip(),
            tool_calls=calls,
            input_tokens=int(usage.get("input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0) or 0)
            + int(usage.get("cache_read_input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0)),
            provider=self.name,
            model=model,
            stop_reason=data.get("stop_reason", ""),
        )
