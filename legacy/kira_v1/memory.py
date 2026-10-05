
import datetime
import uuid

import chromadb
from chromadb.utils import embedding_functions

from audit import log
import config

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="long_term")
observation_collection = client.get_or_create_collection(name="observation")

ef = embedding_functions.DefaultEmbeddingFunction()


def add_memory(text: str, metadata: dict):
    doc_id = str(uuid.uuid4())
    try:
        collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id],
        )
    except Exception as e:
        log("memory_add_error", {"error": str(e)})
        return None
    return doc_id


def _simple_summary(text: str, max_chars: int):
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def add_dialogue(question: str, response: str, summary: str | None = None):
    ts = datetime.datetime.now().isoformat()
    if summary is None:
        summary = _simple_summary(response, config.MEMORY_SUMMARY_MAX_CHARS)
    document = f"Q: {question}\nR: {response}\nSummary: {summary}"
    metadata = {
        "type": "dialogue",
        "question": question,
        "response": response,
        "summary": summary,
        "timestamp": ts,
    }
    return add_memory(document, metadata)


def search_memory(query: str, n=3):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas"],
        )
        return results
    except Exception as e:
        log("memory_query_error", {"error": str(e)})
        return {}


def prune_memory(max_items: int):
    try:
        total = collection.count()
        if total <= max_items:
            return 0
        results = collection.get(include=["ids", "metadatas"])
        ids = results.get("ids", [])
        metas = results.get("metadatas", [])
        items = []
        for doc_id, meta in zip(ids, metas):
            ts = ""
            if meta:
                ts = meta.get("timestamp", "")
            items.append((doc_id, ts))
        items.sort(key=lambda x: x[1] or "")
        to_delete = [doc_id for doc_id, _ in items[: total - max_items]]
        if to_delete:
            collection.delete(ids=to_delete)
        log("memory_prune", {"before": total, "after": total - len(to_delete)})
        return len(to_delete)
    except Exception as e:
        log("memory_prune_error", {"error": str(e)})
        return 0


def search_observation(query: str, n: int = 5):
    try:
        results = observation_collection.query(
            query_texts=[query],
            n_results=n,
            where={"validated": True},
            include=["documents", "metadatas"],
        )
    except Exception:
        results = observation_collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas"],
        )
    documents = results.get("documents", [[]])[0] if results else []
    metadatas = results.get("metadatas", [[]])[0] if results else []

    formatted = []
    for doc, meta in zip(documents, metadatas):
        meta = meta or {}
        if meta.get("validated") is False or meta.get("rejected") is True:
            continue
        score = meta.get("source_score")
        score_text = f" | Fiabilite : {score:.2f}" if isinstance(score, (int, float)) else ""
        citation = (
            f"[Observation | Source : {meta.get('source_url', '')} | "
            f"Date : {meta.get('fetch_date', '')}{score_text}]"
        )
        formatted.append(f"{str(doc).strip()}\n{citation}")
    return "\n\n".join(formatted)
