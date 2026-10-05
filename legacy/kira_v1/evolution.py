import datetime
import json
import os

from audit import log
from config import PROPRIETAIRE
from memory import search_memory

EVOLUTION_DIR = "evolutions"
AUTO_DIR = os.path.join(EVOLUTION_DIR, "auto")
COUNTER_FILE = "evolution_counter.txt"


def ensure_dirs():
    os.makedirs(AUTO_DIR, exist_ok=True)


def get_interaction_count():
    if os.path.exists(COUNTER_FILE):
        try:
            with open(COUNTER_FILE, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except (ValueError, OSError):
            return 0
    return 0


def increment_interaction():
    count = get_interaction_count() + 1
    with open(COUNTER_FILE, "w", encoding="utf-8") as f:
        f.write(str(count))
    return count


def _audit_stats(limit: int):
    lines = []
    if os.path.exists("audit.log"):
        with open("audit.log", "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit * 3 :]

    cloud_count = sum("cloud_query_success" in line for line in lines)
    local_count = sum("local_query" in line for line in lines)
    error_count = sum(
        ("cloud_query_error" in line) or ("Erreur" in line) for line in lines
    )

    recent_events = []
    for line in lines[-5:]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        recent_events.append(
            f"{entry.get('timestamp', '')} | {entry.get('action', '')}"
        )

    return cloud_count, local_count, error_count, recent_events


def _save_auto_record(filename: str, content: str):
    ensure_dirs()
    with open(os.path.join(AUTO_DIR, filename), "w", encoding="utf-8") as f:
        f.write(content)


def analysis_has_opportunity(analysis: str):
    negative_markers = [
        "aucune opportunite",
        "aucune opportunité",
        "aucune amelioration",
        "aucune amélioration",
        "rien a signaler",
        "rien à signaler",
        "pas de proposition",
    ]
    keywords = [
        "opportunite",
        "opportunité",
        "amelioration",
        "amélioration",
        "proposition",
        "suggere",
        "suggère",
        "recommande",
        "recommandation",
        "priorise",
        "priorisé",
        "axe d'evolution",
        "axe d'évolution",
        "priorite",
        "priorité",
    ]
    lowered = analysis.lower()
    if any(k in lowered for k in negative_markers):
        return False
    return any(k in lowered for k in keywords)


def analyze_recent(limit: int = 30):
    """Analyse les dernières entrées du journal d'audit et de la mémoire"""
    ensure_dirs()

    recent_mem = search_memory("dialogue", n=limit)
    docs = []
    documents = recent_mem.get("documents")
    if documents and isinstance(documents, list):
        docs = documents[0] if documents else []
    dialogues = "\n---\n".join(str(doc) for doc in docs if doc)

    cloud_count, local_count, error_count, recent_events = _audit_stats(limit)

    analysis_prompt = f"""
Tu es KIRA en mode méta-analyse renforcé.
Propriétaire : {PROPRIETAIRE}

Contexte d'utilisation :
- Requêtes Groq : {cloud_count}
- Requêtes locales : {local_count}
- Erreurs détectées : {error_count}

Analyse les {limit} dernières interactions et identifie clairement 3 catégories :
1. Performances (latence, erreurs, fallback fréquent)
2. Cohérence (réponses contradictoires, alignement principes)
3. Nouvelles capacités (modules, règles, orchestration, prompts)

Propose 3 axes d'évolution priorisés avec impacts estimés.
Réponds en français, structuré, précis. Ne propose jamais d'application automatique.
"""
    full_prompt = analysis_prompt
    if recent_events:
        full_prompt += "\n\nDerniers événements d'audit :\n" + "\n".join(recent_events)
    if dialogues:
        full_prompt += "\n\nExtrait des dialogues récents :\n" + dialogues[:4000]

    from orchestrator import query_models

    responses = query_models(full_prompt)
    if isinstance(responses, str):
        analysis = responses
    else:
        analysis = max(responses.values(), key=len) if responses else "Analyse impossible."

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_auto_record(f"analyse_{timestamp}.txt", analysis)
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "limit": limit,
        "dialogues_count": len(docs),
        "groq_usage": cloud_count,
        "local_usage": local_count,
        "errors": error_count,
    }
    _save_auto_record(
        f"report_{timestamp}.json", json.dumps(report, ensure_ascii=False, indent=2)
    )

    log(
        "evolution_auto_analysis",
        {
            "limit": limit,
            "groq_usage": cloud_count,
            "local_usage": local_count,
            "errors": error_count,
        },
    )
    return analysis


def propose_patch_from_analysis(analysis: str, trigger: str = "auto", description: str = ""):
    ensure_dirs()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    proposal_id = f"evolution_{timestamp}_{trigger}"
    proposal_path = os.path.join(EVOLUTION_DIR, proposal_id)
    os.makedirs(proposal_path)

    with open(os.path.join(proposal_path, "analyse_source.txt"), "w", encoding="utf-8") as f:
        f.write(analysis)

    patch_prompt = f"""
Sur la base de cette analyse méta, génère 3 propositions concrètes et indépendantes de modification :
- Fichier concerné
- Diff précis (format unifié)
- Explication détaillée des bénéfices/risques
- Tests recommandés avant application

Analyse :
{analysis}
"""
    from orchestrator import query_models

    responses = query_models(patch_prompt)
    if isinstance(responses, str):
        patch_proposal = responses
    else:
        patch_proposal = max(responses.values(), key=len)

    with open(
        os.path.join(proposal_path, "propositions_patch.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(patch_proposal)

    manifest = {
        "id": proposal_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "trigger": trigger,
        "description": description or "",
    }
    with open(os.path.join(proposal_path, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _save_auto_record(f"proposal_{proposal_id}.txt", patch_proposal)
    _save_auto_record(f"proposal_{proposal_id}.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    log("evolution_proposal_generated", {"proposal_id": proposal_id, "trigger": trigger})
    return proposal_id, patch_proposal
