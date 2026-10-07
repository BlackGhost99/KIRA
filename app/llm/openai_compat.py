"""Fournisseurs compatibles « OpenAI chat completions » : OpenAI, DeepSeek, Groq, Ollama local."""
from __future__ import annotations

import json
import hashlib
from typing import Callable

from .. import config, net
from .errors import http_error
from .base import LLMError, LLMResult, Provider, ToolCall, ToolSpec


NO_VISION_NOTE = "\n[Une image accompagnait ce résultat, mais ce modèle ne sait pas lire les images : dis-le à Brice si elle était nécessaire.]"


def convert_messages(system: list[str], messages: list[dict], vision: bool = False, google: bool = False) -> list[dict]:
    out: list[dict] = []
    sys_text = "\n\n".join(t for t in system if t and t.strip())
    if sys_text:
        out.append({"role": "system", "content": sys_text})
    images: list[dict] = []  # images des résultats d'outils : l'API n'en accepte pas dans un message « tool »

    def flush_images() -> None:
        if images:
            parts: list[dict] = [{"type": "text", "text": "Image(s) renvoyée(s) par l'outil ci-dessus :"}]
            parts += [{"type": "image_url", "image_url": {"url": f"data:{i['mime']};base64,{i['b64']}"}} for i in images]
            out.append({"role": "user", "content": parts})
            images.clear()

    for m in messages:
        role = m["role"]
        if role != "tool":
            flush_images()  # après le dernier résultat d'outil de la série
        if role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            msg: dict = {"role": "assistant", "content": m.get("content") or None}
            calls = m.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                    for tc in calls
                ]
            if calls and google:
                for encoded, tc in zip(msg["tool_calls"], calls):
                    if tc.extra_content:
                        encoded["extra_content"] = tc.extra_content
            if msg["content"] is None and not calls:
                continue
            out.append(msg)
        elif role == "tool":
            content = m.get("content") or "(vide)"
            if m.get("images"):
                if vision:
                    images.extend(m["images"])
                else:
                    content += NO_VISION_NOTE
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": content})
    flush_images()
    return out


class OpenAICompatProvider(Provider):
    """``cfg()`` renvoie (clé, base_url, {tier: modèle}, utilise_max_completion_tokens)."""

    def __init__(self, name: str, cfg: Callable[[], tuple[str, str, dict, bool]], needs_key: bool = True,
                 vision: bool = False):
        self.name = name
        self._cfg = cfg
        self._needs_key = needs_key
        self.vision = vision  # le modèle sait-il lire une image (capture d'écran) ?

    def configured(self) -> bool:
        key, base, models, _ = self._cfg()
        return bool(base) and (bool(key) or not self._needs_key) and bool(models.get("default"))

    def cost_class(self) -> str:
        s = config.settings
        if self.name == "ollama":
            return "self_hosted"
        if self.name == "openrouter" or (self.name == "gemini" and s.gemini_free_tier) or (self.name == "groq" and s.groq_free_tier):
            return "free"
        return "paid"

    def fingerprint(self) -> str:
        return hashlib.sha256(repr(self._cfg()).encode()).hexdigest()

    def model_for(self, tier: str) -> str:
        _, _, models, _ = self._cfg()
        return models.get(tier) or models["default"]

    def complete(self, system, messages, tools, tier, max_tokens) -> LLMResult:
        key, base, _, completion_tokens = self._cfg()
        model = self.model_for(tier)
        body: dict = {"model": model, "messages": convert_messages(system, messages, self.vision, google=self.name == "gemini")}
        body["max_completion_tokens" if completion_tokens else "max_tokens"] = max_tokens
        if self.name == "openrouter":
            if model != "openrouter/free" and not model.endswith(":free"):
                raise LLMError("OpenRouter : seul un modèle gratuit est autorisé", 400, self.name, category="configuration")
            body["provider"] = {"require_parameters": True, "max_price": {"prompt": 0, "completion": 0, "request": 0, "image": 0}}
        if tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        headers = {"content-type": "application/json"}
        if key:
            headers["authorization"] = f"Bearer {key}"
        resp = net.session().post(base.rstrip("/") + "/chat/completions", headers=headers, json=body, allow_redirects=False, timeout=(10, max(5, min(120, config.settings.llm_timeout))))
        if resp.status_code != 200:
            raise http_error(resp, self.name)
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError(f"{self.name} : JSON invalide", provider=self.name) from exc
        try:
            return self._parse_response(data, model)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise LLMError(f"{self.name} : réponse invalide", provider=self.name) from exc

    def _parse_response(self, data, model) -> LLMResult:
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name} : réponse inattendue", None, self.name) from exc
        calls: list[ToolCall] = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args if isinstance(args, dict) else {},
                                  tc.get("extra_content", {}) if self.name == "gemini" else {}))
        if not isinstance(msg.get("content") or "", str) or (not msg.get("content") and not calls):
            raise LLMError(f"{self.name} : réponse vide ou invalide", provider=self.name)
        usage = data.get("usage") or {}
        return LLMResult(
            text=(msg.get("content") or "").strip(),
            tool_calls=calls,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            provider=self.name,
            model=str(data.get("model") or model)[:200],
            stop_reason=choice.get("finish_reason", ""),
        )


def openai_provider() -> OpenAICompatProvider:
    def cfg():
        s = config.settings
        models = {"default": s.openai_model, "fast": s.openai_model_fast or s.openai_model}
        if s.openai_model_deep:
            models["deep"] = s.openai_model_deep
        return s.openai_api_key, s.openai_base_url, models, True

    return OpenAICompatProvider("openai", cfg, vision=True)


def deepseek_provider() -> OpenAICompatProvider:
    def cfg():
        s = config.settings
        return s.deepseek_api_key, s.deepseek_base_url, {"default": s.deepseek_model}, False

    return OpenAICompatProvider("deepseek", cfg)


def groq_provider() -> OpenAICompatProvider:
    def cfg():
        s = config.settings
        return s.groq_api_key, s.groq_base_url, {"default": s.groq_model}, False

    return OpenAICompatProvider("groq", cfg)


def ollama_provider() -> OpenAICompatProvider:
    def cfg():
        s = config.settings
        base = (s.ollama_url.rstrip("/") + "/v1") if s.ollama_url else ""
        return "", base, {"default": s.ollama_model}, False

    return OpenAICompatProvider("ollama", cfg, needs_key=False)


def gemini_provider() -> OpenAICompatProvider:
    def cfg():
        s = config.settings
        return s.gemini_api_key, "https://generativelanguage.googleapis.com/v1beta/openai", {"default": s.gemini_model}, False
    return OpenAICompatProvider("gemini", cfg, vision=True)


def openrouter_provider() -> OpenAICompatProvider:
    def cfg():
        s = config.settings
        return s.openrouter_api_key, "https://openrouter.ai/api/v1", {"default": s.openrouter_model}, False
    return OpenAICompatProvider("openrouter", cfg, vision=True)
