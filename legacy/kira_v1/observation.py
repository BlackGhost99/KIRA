import datetime
import hashlib
import json
import os
import uuid

import feedparser
import requests
from bs4 import BeautifulSoup

from audit import log
from memory import observation_collection
from orchestrator import query_models

# Collection dédiée au canal OBSERVATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OBSERVATION_SOURCES_FILE = os.path.join(BASE_DIR, "observation_sources.txt")  # Une URL par ligne (page web ou RSS)
LAST_RUN_FILE = os.path.join(BASE_DIR, "observation_last_run.json")
LAST_VEILLE_FILE = os.path.join(BASE_DIR, "last_veille_timestamp.txt")
SOURCE_SCORE_DEFAULT = 0.5
MIN_CONTENT_CHARS = 200


def _parse_source_line(line: str):
    if "|" in line:
        url, score_raw = line.split("|", 1)
        url = url.strip()
        score_raw = score_raw.strip()
    else:
        parts = line.split(maxsplit=1)
        url = parts[0].strip() if parts else ""
        score_raw = parts[1].strip() if len(parts) > 1 else ""
    score = SOURCE_SCORE_DEFAULT
    if score_raw:
        try:
            score = float(score_raw)
        except ValueError:
            score = SOURCE_SCORE_DEFAULT
    score = max(0.0, min(1.0, score))
    return {"url": url, "score": score}


def load_sources():
    if not os.path.exists(OBSERVATION_SOURCES_FILE):
        with open(OBSERVATION_SOURCES_FILE, "w", encoding="utf-8") as f:
            f.write("# Liste de sources fiables (une par ligne)\n")
            f.write("# Exemples :\n")
            f.write("# https://arxiv.org/rss/cs\n")
            f.write("# https://news.ycombinator.com/rss\n")
            f.write("# Vous pouvez ajouter un score :\n")
            f.write("# https://arxiv.org/rss/cs | 0.9\n")
        return []
    with open(OBSERVATION_SOURCES_FILE, "r", encoding="utf-8") as f:
        sources = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            source = _parse_source_line(line)
            if source["url"]:
                sources.append(source)
        return sources


def fetch_rss(url: str):
    feed = feedparser.parse(url)
    entries = []
    for entry in feed.entries[:10]:  # Limite à 10 derniers items
        entries.append(
            {
                "title": entry.get("title", ""),
                "link": entry.get("link", url),
                "published": entry.get("published", ""),
                "summary": entry.get("summary", ""),
            }
        )
    return entries


def fetch_page(url: str):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        text = response.text
        if "xml" in content_type or text.lstrip().startswith("<?xml"):
            try:
                soup = BeautifulSoup(text, "xml")
            except Exception:
                soup = BeautifulSoup(text, "html.parser")
        else:
            soup = BeautifulSoup(text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator="\n")
        return text[:10000]  # Limite pour éviter overflow
    except Exception as e:
        return f"[Erreur fetch] {str(e)}"


def _save_last_run(payload: dict):
    with open(LAST_RUN_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_last_run():
    if not os.path.exists(LAST_RUN_FILE):
        return {}
    try:
        with open(LAST_RUN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def get_last_veille_time():
    if os.path.exists(LAST_VEILLE_FILE):
        with open(LAST_VEILLE_FILE, "r", encoding="utf-8") as f:
            try:
                return datetime.datetime.fromisoformat(f.read().strip())
            except (ValueError, OSError):
                return datetime.datetime.min
    return datetime.datetime.min


def update_last_veille_time():
    with open(LAST_VEILLE_FILE, "w", encoding="utf-8") as f:
        f.write(datetime.datetime.now().isoformat())


def should_trigger_auto_veille():
    import config

    if not config.OBSERVATION_AUTO_ENABLED:
        return False
    last_time = get_last_veille_time()
    delta = datetime.datetime.now() - last_time
    return delta >= datetime.timedelta(hours=config.OBSERVATION_AUTO_INTERVAL_HOURS)


def observe_source(url: str, source_score: float):
    if url.endswith(".rss") or "rss" in url.lower() or url.endswith(".xml"):
        items = fetch_rss(url)
    else:
        items = [{"title": "Page complète", "content": fetch_page(url)}]

    results = []
    ids = []
    for item in items:
        content = item.get("summary") or item.get("content", "")
        if not content or content.startswith("[Erreur"):
            continue
        if len(content) < MIN_CONTENT_CHARS:
            continue

        content_hash = hashlib.sha256(
            (item.get("title", "") + item.get("link", url) + content[:1000]).encode(
                "utf-8"
            )
        ).hexdigest()
        try:
            existing = observation_collection.get(where={"content_hash": content_hash})
            if existing.get("ids"):
                continue
        except Exception:
            pass

        summary_prompt = f"""
Résume ce contenu en français de manière neutre et objective.
Si tu identifies un signal faible ou une tendance, explicite-le avec :
"Observation interprétée : ..."
Sinon écris : "Observation interprétée : Aucune."
Contenu :
{content[:8000]}
"""
        responses = query_models(summary_prompt)
        if isinstance(responses, str):
            summary = responses
        else:
            summary = max(responses.values(), key=len)

        published = item.get("published") or datetime.datetime.now().isoformat()
        full_text = (
            f"Titre : {item.get('title', 'Sans titre')}\n"
            f"Date : {published}\n"
            f"Résumé observé :\n{summary}\n\n"
            f"Extrait brut :\n{content}"
        )

        doc_id = str(uuid.uuid4())
        observation_collection.add(
            documents=[full_text],
            metadatas=[
                {
                    "source_url": item.get("link", url),
                    "type": "observation",
                    "fetch_date": datetime.datetime.now().isoformat(),
                    "original_url": url,
                    "validated": False,
                    "excerpt": content[:500],
                    "content_hash": content_hash,
                    "source_score": source_score,
                }
            ],
            ids=[doc_id],
        )

        ids.append(doc_id)
        results.append(
            {
                "title": item.get("title", "Page"),
                "summary": summary,
                "source": item.get("link", url),
                "score": source_score,
            }
        )

    log("observation_veille", {"url": url, "items": len(results)})
    return results, ids


def observe_all(propose_validation: bool = True):
    sources = load_sources()
    if not sources:
        return "Aucune source configurée. Ajoutez des URLs dans observation_sources.txt."

    all_results = []
    all_ids = []
    for source in sources:
        url = source["url"]
        score = source["score"]
        results, ids = observe_source(url, score)
        all_results.extend(results)
        all_ids.extend(ids)

    _save_last_run(
        {
            "timestamp": datetime.datetime.now().isoformat(),
            "sources": [s["url"] for s in sources],
            "ids": all_ids,
        }
    )
    update_last_veille_time()

    if propose_validation and all_results:
        proposal = "\n\n=== PROPOSITION D'INTEGRATION OBSERVATION ===\n"
        proposal += (
            f"Veille effectuée sur {len(sources)} sources "
            f"({datetime.datetime.now().isoformat()}).\n"
        )
        proposal += "Résumés proposés :\n"
        sorted_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)
        for r in sorted_results[:10]:
            proposal += (
                f"- [{r['title']}] {r['summary'][:200]}...\n"
                f"Source : {r['source']} | Fiabilité : {r.get('score', 0):.2f}\n\n"
            )
        proposal += (
            "KIRA : Ces observations sont stockées temporairement. "
            "Confirmez !observation validate pour intégration définitive."
        )
        return proposal

    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_results


def validate_last():
    payload = _load_last_run()
    ids = payload.get("ids", [])
    if not ids:
        return "Aucune observation récente à valider."

    metadatas = []
    for _ in ids:
        metadatas.append({"validated": True})
    try:
        observation_collection.update(ids=ids, metadatas=metadatas)
    except Exception:
        # Fallback: réécrit les documents avec validated=True
        results = observation_collection.get(ids=ids)
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        updated_metas = []
        for meta in metas:
            meta = meta or {}
            meta["validated"] = True
            updated_metas.append(meta)
        observation_collection.upsert(ids=ids, documents=docs, metadatas=updated_metas)

    log("observation_validate", {"count": len(ids)})
    return f"{len(ids)} observations validées."


def validate_pending():
    try:
        items = observation_collection.get(where={"validated": False})
    except Exception:
        items = observation_collection.get()
        ids = []
        for doc_id, meta in zip(items.get("ids", []), items.get("metadatas", [])):
            if meta and meta.get("validated") is False:
                ids.append(doc_id)
        items = {"ids": ids}

    ids = items.get("ids", [])
    if not ids:
        return "Aucune observation en attente de validation."

    metadatas = [{"validated": True} for _ in ids]
    try:
        observation_collection.update(ids=ids, metadatas=metadatas)
    except Exception:
        results = observation_collection.get(ids=ids)
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        updated_metas = []
        for meta in metas:
            meta = meta or {}
            meta["validated"] = True
            updated_metas.append(meta)
        observation_collection.upsert(ids=ids, documents=docs, metadatas=updated_metas)

    log("observation_validated", {"count": len(ids)})
    return f"{len(ids)} observations validées et intégrées définitivement."


def auto_validate_by_score(min_score: float = 0.8):
    try:
        items = observation_collection.get(where={"validated": False})
    except Exception:
        items = observation_collection.get()
    ids = []
    metas = items.get("metadatas", [])
    for doc_id, meta in zip(items.get("ids", []), metas):
        meta = meta or {}
        score = meta.get("source_score", 0)
        if meta.get("validated") is False and isinstance(score, (int, float)) and score >= min_score:
            ids.append(doc_id)
    if not ids:
        return 0
    metadatas = [{"validated": True} for _ in ids]
    try:
        observation_collection.update(ids=ids, metadatas=metadatas)
    except Exception:
        results = observation_collection.get(ids=ids)
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        updated_metas = []
        for meta in metas:
            meta = meta or {}
            meta["validated"] = True
            updated_metas.append(meta)
        observation_collection.upsert(ids=ids, documents=docs, metadatas=updated_metas)
    log("observation_auto_validated", {"count": len(ids), "min_score": min_score})
    return len(ids)


def reject_last():
    payload = _load_last_run()
    ids = payload.get("ids", [])
    if not ids:
        return "Aucune observation récente à rejeter."

    try:
        observation_collection.delete(ids=ids)
    except Exception:
        metadatas = [{"rejected": True, "validated": False} for _ in ids]
        try:
            observation_collection.update(ids=ids, metadatas=metadatas)
        except Exception:
            results = observation_collection.get(ids=ids)
            docs = results.get("documents", [])
            observation_collection.upsert(ids=ids, documents=docs, metadatas=metadatas)

    log("observation_reject", {"count": len(ids)})
    return f"{len(ids)} observations rejetées."
