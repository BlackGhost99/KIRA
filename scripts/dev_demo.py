"""Démonstration locale de KIRA, SANS clé d'API : une fausse IA répond à la place des vrais modèles.

Sert à voir et tester l'interface. Mot de passe : demo.   Lancement :  python scripts/dev_demo.py [port]
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = Path(tempfile.mkdtemp(prefix="kira_demo_"))
os.environ.update(
    DATABASE_URL=f"sqlite:///{DATA / 'demo.db'}",
    OWNER_PASSWORD="demo",
    COOKIE_SECURE="false",
    SECRET_KEY="demo-secret-key-not-for-production",
    CRON_TOKEN="demo-cron",
    OWNER_NAME="Brice",
    TIMEZONE="Africa/Libreville",
)
for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "GITHUB_TOKEN", "TAVILY_API_KEY"):
    os.environ.pop(key, None)

from app import db, memory  # noqa: E402
from app.llm import router as llm_router  # noqa: E402
from app.llm.base import LLMResult, Provider, ToolCall  # noqa: E402

PLOT = """import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0, 4*np.pi, 400)
plt.plot(t, np.cos(t), label='x(t) = cos(ωt)')
plt.plot(t, -np.sin(t), label='v(t)')
plt.xlabel('temps (s)'); plt.ylabel('amplitude'); plt.legend(); plt.title('Oscillateur harmonique')
"""

PHYSICS = """Bonne question, et on va y aller doucement : **l'énergie d'un système, c'est sa capacité à faire changer les choses.**

### L'intuition
Une balle qui tombe convertit de la hauteur en vitesse. Rien ne se perd, tout se transforme.

### Le formalisme
Pour une masse $m$ à la hauteur $h$, l'énergie potentielle vaut $E_p = mgh$, et l'énergie cinétique $E_c = \\frac{1}{2}mv^2$. Leur somme est conservée :

$$E_c + E_p = \\frac{1}{2}mv^2 + mgh = \\text{constante}$$

| Grandeur | Symbole | Unité |
|---|---|---|
| Masse | $m$ | kg |
| Vitesse | $v$ | m/s |
| Énergie | $E$ | J |

Un petit calcul pour fixer les idées :

```python
m, h, g = 2.0, 5.0, 9.81
print(m * g * h)  # 98.1 J
```

Veux-tu qu'on vérifie ça avec une simulation, ou qu'on passe à la conservation de l'énergie en mécanique lagrangienne ?"""

GENERIC = """Je suis là. Voici comment je te propose de procéder :

1. Tu me dis où tu en es sur le sujet.
2. Je pose trois questions pour repérer ce qui manque.
3. On construit ensuite pas à pas, avec des exemples concrets.

> Tu n'as pas besoin de tout savoir avant de commencer.

Par quoi veux-tu démarrer ?"""


class DemoProvider(Provider):
    name = "demo"

    def configured(self) -> bool:
        return True

    def model_for(self, tier: str) -> str:
        return {"fast": "démo-rapide", "deep": "démo-profonde"}.get(tier, "démo")

    def complete(self, system, messages, tools, tier, max_tokens):
        last = messages[-1]
        res = LLMResult(provider=self.name, model=self.model_for(tier), input_tokens=900, output_tokens=300)
        if last["role"] == "tool" and last.get("name") in ("device_action", "devices", "device_result"):
            body = last["content"]
            if "Refusé" in body or "refusé" in body:
                res.text = "D'accord, je n'y touche pas. " + body.splitlines()[-1][:200]
            elif "En attente" in body:
                res.text = "J'ai besoin de ton accord pour ça : la demande est affichée en bas de l'écran, avec le détail exact."
            else:
                res.text = "C'est fait. Voici ce que la machine m'a répondu :\n\n```\n" + "\n".join(body.splitlines()[2:14]) + "\n```"
            return res
        if last["role"] == "tool":
            image = re.search(r"!\[[^\]]*\]\(/api/files/[0-9a-f]{32}\)", last["content"])
            res.text = "Voilà la solution d'un oscillateur harmonique : la position est en phase avec $\\cos(\\omega t)$.\n\n" + (
                image.group(0) if image else "(pas de graphique)"
            ) + "\n\nLa vitesse est déphasée d'un quart de période : quand $x$ est maximal, $v$ s'annule."
            return res
        text = last["content"].lower()
        if tools and any(t.name == "device_action" for t in tools) and any(w in text for w in ("dossier", "fichier", "machine", "pc")):
            res.tool_calls = [ToolCall("d1", "device_action", {"action": "list_dir", "args": {"path": "."}})]
        elif tools and any(t.name == "device_action" for t in tools) and any(w in text for w in ("écris", "ecris", "note-moi")):
            res.tool_calls = [ToolCall("d2", "device_action", {"action": "write_file", "args": {"path": "notes/idee.md", "content": "# Idée\n\nRevoir le théorème de Noether demain matin.\n"}})]
        elif tools and any(t.name == "device_action" for t in tools) and any(w in text for w in ("commande", "lance", "système")):
            res.tool_calls = [ToolCall("d3", "device_action", {"action": "run_command", "args": {"command": "uname -a && date && ls -la"}})]
        elif any(w in text for w in ("trace", "graphique", "courbe")):
            res.tool_calls = [ToolCall("c1", "python", {"code": PLOT})]
        elif any(w in text for w in ("énergie", "energie", "formule", "physique")):
            res.text = PHYSICS
        elif "erreur" in text:
            from app.llm.base import LLMUnavailable
            raise LLMUnavailable("Aucun fournisseur d'IA n'a pu répondre. Vérifie tes clés d'API.")
        else:
            res.text = GENERIC
        return res


def seed() -> None:
    d = db.get_db()
    memory.add("profile", "Brice se dit moins à l'aise en physique ; il veut combler ses lacunes en mécanique.", "physique")
    memory.add("profile", "Brice habite au Gabon et travaille depuis un PC de bureau (i7) : tout doit tourner dans le cloud.", "contexte")
    memory.add("fact", "Brice prépare des expériences scientifiques et veut un partenaire de recherche.", "projet")
    memory.add("knowledge", "Principe de moindre action : la trajectoire réelle rend stationnaire l'intégrale de L = T − V.", "physique", source="https://www.quantamagazine.org/")
    memory.add("note", "Revoir les équations de Lagrange avant la prochaine séance.", "à faire")
    src = d.insert("veille_sources", {"url": "https://www.quantamagazine.org/feed", "name": "Quanta Magazine", "kind": "rss", "score": 0.9, "active": 1})
    d.insert("veille_sources", {"url": "https://phys.org/rss-feed/", "name": "Phys.org", "kind": "rss", "score": 0.6, "active": 1, "last_error": "HTTP 503"})
    for i, (title, summary, topic, score) in enumerate([
        ("Pourquoi les symétries gouvernent la physique", "Le théorème de Noether relie chaque symétrie d'un système à une grandeur conservée : énergie, quantité de mouvement, moment cinétique.", "physique", 0.92),
        ("Apprendre les équations différentielles par l'exemple", "Un cours en ligne qui part de phénomènes concrets (ressort, circuit RC) avant d'introduire le formalisme.", "mathématiques", 0.74),
        ("Les nouveautés de Python 3.14 pour le calcul scientifique", "Meilleures performances et nouveau profileur ; utile pour les simulations numériques.", "informatique", 0.55),
    ]):
        d.insert("veille_items", {"source_id": src, "title": title, "url": "https://exemple.org/a%d" % i, "summary": summary,
                                  "topic": topic, "level": "débutant", "score": score, "status": "pending",
                                  "content_hash": "demo%d" % i, "created_at": db.now_iso()})
    d.insert("proposals", {
        "title": "Expliquer plus lentement les formules",
        "rationale": "Tu as signalé deux fois que mes explications de formules allaient trop vite. Je propose d'ajouter une étape « lecture du symbole » avant chaque formule.",
        "trigger_kind": "manual", "changes": "[]", "tests_ok": 1, "status": "draft",
        "diff": "--- a/app/agent.py\n+++ b/app/agent.py\n@@ -31,3 +31,4 @@\n - Écris les formules en LaTeX.\n+ - Avant chaque formule, nomme chaque symbole en une phrase.\n",
        "created_at": db.now_iso(), "updated_at": db.now_iso()})


def start_core(port: int):
    """Relie une vraie machine de démonstration (le vrai agent, dans un dossier jetable) au serveur de démo."""
    from app import devices

    home = DATA / "machine"
    root = home / "KIRA"
    (root / "cours").mkdir(parents=True)
    (root / "cours" / "mecanique.md").write_text("# Mécanique\n\nL'énergie se conserve.\n", encoding="utf-8")
    (root / "cours" / "maths.txt").write_text("dérivées, intégrales\n", encoding="utf-8")
    env = dict(os.environ, HOME=str(home), KIRA_CORE_HOME=str(home / ".kira-core"), USERPROFILE=str(home), NO_PROXY="127.0.0.1,localhost")
    url = f"http://127.0.0.1:{port}"
    agent = ROOT / "core" / "kira_core.py"
    code = devices.create_pair_code()["code"]
    subprocess.run([sys.executable, str(agent), "pair", "--server", url, "--code", code, "--name", "PC de démonstration", "--full"],
                   env=env, check=True, capture_output=True)
    return subprocess.Popen([sys.executable, str(agent), "run"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    import uvicorn

    from app.main import app

    db.init_db()
    seed()
    llm_router.set_router(llm_router.Router(providers=[DemoProvider()]))
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"KIRA (démo) sur http://127.0.0.1:{port}   mot de passe : demo   données : {DATA}")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.1)
    core = None
    if os.environ.get("KIRA_DEMO_NO_CORE") != "1":
        core = start_core(port)
        print("Machine de démonstration reliée (PC de démonstration).")
    try:
        thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        if core:
            core.terminate()
