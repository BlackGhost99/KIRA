"""Registre des outils que l'agent peut appeler."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .. import audit
from ..llm.base import ToolSpec

MAX_RESULT_CHARS = 9000


@dataclass
class ToolContext:
    conversation_id: str = ""
    files: list[dict] = field(default_factory=list)  # fichiers produits pendant le tour
    tainted: bool = False       # du contenu venu d'Internet a été lu pendant ce tour : il peut contenir un piège
    private_data: bool = False  # des données de la machine de Brice ont été lues pendant ce tour : elles ne sortent pas
    images: list[dict] = field(default_factory=list)  # [{"mime", "b64"}] à montrer au modèle avec le prochain résultat


@dataclass
class Tool:
    spec: ToolSpec
    fn: Callable[[dict, ToolContext], str]
    label: str  # texte affiché pendant l'exécution
    audit_args: Callable[[dict], dict] | None = None  # ce qui va au journal (par défaut : arguments raccourcis)


REGISTRY: dict[str, Tool] = {}


def register(name: str, description: str, parameters: dict, label: str, audit_args: Callable[[dict], dict] | None = None):
    def deco(fn: Callable[[dict, ToolContext], str]):
        REGISTRY[name] = Tool(ToolSpec(name, description, parameters), fn, label, audit_args)
        return fn

    return deco


def specs() -> list[ToolSpec]:
    return [t.spec for t in REGISTRY.values()]


def label_for(name: str) -> str:
    tool = REGISTRY.get(name)
    return tool.label if tool else name


def run(name: str, args: dict, ctx: ToolContext) -> str:
    """Exécute un outil. Ne lève jamais : l'erreur est renvoyée au modèle sous forme de texte."""
    tool = REGISTRY.get(name)
    if tool is None:
        return f"Erreur : outil inconnu « {name} »."
    try:
        out = tool.fn(args or {}, ctx)
    except Exception as exc:  # noqa: BLE001
        audit.log("tool_error", {"tool": name, "error": f"{type(exc).__name__}: {exc}"[:300]})
        return f"Erreur dans l'outil {name} : {type(exc).__name__}: {exc}"
    out = out if isinstance(out, str) else str(out)
    if len(out) > MAX_RESULT_CHARS:
        out = out[:MAX_RESULT_CHARS] + f"\n[... {len(out) - MAX_RESULT_CHARS} caractères coupés]"
    audit.log("tool_call", {"tool": name, "args": (tool.audit_args or _short)(args or {})})
    return out


def _short(args):
    if isinstance(args, str):
        return args[:200] + "…" if len(args) > 200 else args
    if isinstance(args, dict):
        return {k: _short(v) for k, v in list(args.items())[:20]}
    if isinstance(args, list):
        return [_short(v) for v in args[:20]]
    return args
