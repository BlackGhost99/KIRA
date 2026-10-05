"""Outils de mémoire : retenir un fait durable, retrouver un souvenir ou un ancien échange."""
from __future__ import annotations

from .. import memory
from .registry import ToolContext, register


@register(
    "remember",
    "Enregistre dans la mémoire durable un fait utile pour plus tard. Utilise kind='profile' pour ce que tu apprends "
    "sur Brice (niveau dans une matière, lacunes, objectifs, préférences, projets) et kind='fact' pour le reste. "
    "N'enregistre pas les détails passagers.",
    {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Une phrase claire et autonome."},
            "kind": {"type": "string", "enum": ["profile", "fact", "note"]},
            "tags": {"type": "string", "description": "Mots-clés séparés par des virgules (facultatif)."},
        },
        "required": ["content"],
    },
    "Mémorisation",
)
def remember(args: dict, ctx: ToolContext) -> str:
    item = memory.add(args.get("kind") or "fact", args.get("content", ""), args.get("tags", ""), source="chat")
    return f"Enregistré (#{item['id']}, {item['kind']})."


@register(
    "recall",
    "Cherche dans la mémoire durable et dans les anciens échanges de Brice avec KIRA.",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    "Recherche dans la mémoire",
)
def recall(args: dict, ctx: ToolContext) -> str:
    query = args.get("query", "")
    mem = memory.search(query, limit=6)
    msgs = memory.search_messages(query, limit=4)
    lines: list[str] = []
    for m in mem:
        if m["kind"] == "knowledge":
            ctx.tainted = True  # résumé d'un article lu sur Internet
        lines.append(f"[mémoire #{m['id']} · {m['kind']}] {m['content'][:400]}")
    for m in msgs:
        who = "Brice" if m["role"] == "user" else "KIRA"
        lines.append(f"[échange du {m['created_at'][:10]} · {who}] {m['content'][:300]}")
    return "\n".join(lines) if lines else "Rien trouvé."
