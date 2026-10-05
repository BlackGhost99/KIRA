"""Types communs aux fournisseurs d'IA : un seul format de messages pour tous."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # schéma JSON des arguments


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResult:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""
    model: str = ""
    stop_reason: str = ""


class LLMError(Exception):
    def __init__(self, message: str, status: int | None = None, provider: str = ""):
        super().__init__(message)
        self.status = status
        self.provider = provider


class LLMUnavailable(Exception):
    """Aucun fournisseur n'a pu répondre."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []


class BudgetExceeded(Exception):
    """Le plafond quotidien de jetons est atteint."""


class Provider:
    """Un fournisseur d'IA. Les messages suivent le format neutre de KIRA :

    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}
    """

    name = "base"

    def configured(self) -> bool:
        raise NotImplementedError

    def model_for(self, tier: str) -> str:
        raise NotImplementedError

    def complete(self, system: list[str], messages: list[dict], tools: list[ToolSpec] | None,
                 tier: str, max_tokens: int) -> LLMResult:
        raise NotImplementedError
