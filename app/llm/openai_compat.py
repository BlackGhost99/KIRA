"""Fournisseurs compatibles « OpenAI chat completions » : OpenAI, DeepSeek, Groq, Ollama local."""
from __future__ import annotations

import json
from typing import Callable

from .. import config, net
from .base import LLMError, LLMResult, Provider, ToolCall, ToolSpec


NO_VISION_NOTE = "\n[Une image accompagnait ce résultat, mais ce modèle ne sait pas lire les images : dis-le à Brice si elle était nécessaire.]"


def convert_messages(system: list[str], messages: list[dict], vision: bool = False) -> list[dict]:
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

    def model_for(self, tier: str) -> str:
        _, _, models, _ = self._cfg()
        return models.get(tier) or models["default"]

    def complete(self, system, messages, tools, tier, max_tokens) -> LLMResult:
        key, base, _, completion_tokens = self._cfg()
        model = self.model_for(tier)
        body: dict = {"model": model, "messages": convert_messages(system, messages, self.vision)}
        body["max_completion_tokens" if completion_tokens else "max_tokens"] = max_tokens
        if tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        headers = {"content-type": "application/json"}
        if key:
            headers["authorization"] = f"Bearer {key}"
        resp = net.session().post(base.rstrip("/") + "/chat/completions", headers=headers, json=body, timeout=(300 if max_tokens > 6000 else 150))
        if resp.status_code != 200:
            raise LLMError(f"{self.name} HTTP {resp.status_code} : {resp.text[:300]}", resp.status_code, self.name)
        data = resp.json()
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
            calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args if isinstance(args, dict) else {}))
        usage = data.get("usage") or {}
        return LLMResult(
            text=(msg.get("content") or "").strip(),
            tool_calls=calls,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            provider=self.name,
            model=model,
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
