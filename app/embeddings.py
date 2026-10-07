"""Embeddings via un endpoint cloud choisi par le propriétaire (aucun modèle local)."""
from __future__ import annotations

import hashlib
import math
from urllib.parse import urlsplit

import requests

from . import config


class EmbeddingUnavailable(RuntimeError):
    pass


def configured() -> bool:
    return bool(config.settings.embedding_url and config.settings.embedding_api_key)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate(vector, dimensions: int) -> list[float]:
    if not isinstance(vector, list) or len(vector) != dimensions or not 1 <= dimensions <= 4096:
        raise EmbeddingUnavailable("Dimension d'embedding incompatible.")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vector):
        raise EmbeddingUnavailable("Embedding invalide.")
    values = [float(v) for v in vector]
    if not all(math.isfinite(v) for v in values):
        raise EmbeddingUnavailable("Embedding non fini.")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0:
        raise EmbeddingUnavailable("Embedding de norme invalide.")
    return [v / norm for v in values]


def embed(text: str) -> list[float]:
    s = config.settings
    if not configured():
        raise EmbeddingUnavailable("Service d'embeddings non configuré.")
    url = urlsplit(s.embedding_url)
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise EmbeddingUnavailable("L'endpoint d'embeddings doit utiliser HTTPS.")
    try:
        # Endpoint de configuration de confiance, jamais une URL fournie par un outil ou par le LLM.
        with requests.Session() as session:
            session.trust_env = False
            with session.post(
                s.embedding_url, json={"input": text[:8000], "model": s.embedding_model},
                headers={"Authorization": "Bearer " + s.embedding_api_key},
                timeout=max(1, min(s.embedding_timeout, 30)), allow_redirects=False, stream=True,
            ) as response:
                if response.status_code != 200:
                    raise EmbeddingUnavailable("Service d'embeddings indisponible.")
                body = bytearray()
                for chunk in response.iter_content(8192):
                    body.extend(chunk)
                    if len(body) > 200_000:
                        raise EmbeddingUnavailable("Réponse d'embedding trop volumineuse.")
                import json
                data = json.loads(body)
                if not isinstance(data, dict) or ("model" in data and data["model"] != s.embedding_model):
                    raise EmbeddingUnavailable("Modèle d'embedding incompatible.")
                vector = data.get("embedding")  # contrat de la fonction Supabase optionnelle
                if vector is None:  # contrat compatible OpenAI (Cloudflare et autres endpoints)
                    rows = data.get("data")
                    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
                        raise EmbeddingUnavailable("Réponse d'embedding incompatible.")
                    vector = rows[0].get("embedding")
                return validate(vector, s.embedding_dimensions)
    except (requests.RequestException, ValueError, TypeError, OverflowError) as exc:
        # Ne pas propager une URL, un contenu ou un en-tête sensible dans les journaux.
        raise EmbeddingUnavailable("Service d'embeddings indisponible.") from None
