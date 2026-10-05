"""Outil de calcul : du vrai Python (numpy, scipy, sympy, matplotlib) au lieu de calculer de tête."""
from __future__ import annotations

from .. import files, sandbox
from .registry import ToolContext, register


@register(
    "python",
    "Exécute du code Python pour calculer, simuler, résoudre symboliquement (sympy) ou tracer un graphique "
    "(matplotlib). numpy, scipy, sympy et matplotlib sont disponibles. Affiche les résultats avec print(). "
    "Les graphiques sont enregistrés automatiquement (plt.show() ou figure ouverte à la fin) et l'outil renvoie la "
    "ligne Markdown à copier dans ta réponse pour les montrer. 30 secondes maximum, pas de fichiers persistants.",
    {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Le code Python à exécuter."}},
        "required": ["code"],
    },
    "Calcul en Python",
)
def python_tool(args: dict, ctx: ToolContext) -> str:
    code = args.get("code", "")
    if not isinstance(code, str) or not code.strip():
        return "Erreur : le code est vide."
    if len(code) > 30000:
        return "Erreur : code trop long (30 000 caractères maximum)."
    res = sandbox.run_python(code)
    parts: list[str] = []
    if res["timed_out"]:
        parts.append("Délai dépassé : le calcul a été interrompu.")
    if res["stdout"].strip():
        parts.append("Sortie :\n" + res["stdout"].rstrip())
    if res["stderr"].strip():
        parts.append(("Erreur :\n" if not res["ok"] else "Avertissements :\n") + res["stderr"].rstrip())
    for i, (name, data) in enumerate(res["files"], 1):
        fid = files.save(name, "image/png", data)
        ctx.files.append({"id": fid, "name": name, "url": files.url(fid)})
        parts.append(f"Graphique {i} enregistré. Pour l'afficher, écris : ![Graphique {i}]({files.url(fid)})")
    if not parts:
        parts.append("Le code s'est exécuté sans rien afficher (utilise print() pour voir un résultat).")
    return "\n\n".join(parts)
