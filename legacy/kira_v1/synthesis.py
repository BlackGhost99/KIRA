import re

import config

ERROR_PREFIXES = ("[Erreur", "[Erreur locale", "[Erreur Groq")


def _normalize(text: str):
    return re.sub(r"\s+", " ", text or "").strip()


def score_response(text: str):
    if not text:
        return -1000
    score = 0
    cleaned = _normalize(text)
    length = len(cleaned)

    if cleaned.startswith(ERROR_PREFIXES):
        score -= 2000

    if length < config.SYNTHESIS_MIN_LEN:
        score -= 200
    score += min(length, config.SYNTHESIS_MAX_LEN)

    if "\n- " in text or "\n1." in text or "\n2." in text:
        score += 80
    if "Conclusion" in text or "Synthese" in text or "Synthèse" in text:
        score += 40
    if "Je ne sais pas" in text or "je ne sais pas" in text:
        score -= 100

    return score


def select_best_response(responses: dict):
    best_text = None
    best_score = -10_000
    for _, text in responses.items():
        s = score_response(text)
        if s > best_score:
            best_score = s
            best_text = text
    return best_text or ""


def build_consensus_prompt(question: str, responses: dict):
    blocks = []
    for name, text in responses.items():
        blocks.append(f"[{name}]\n{text}")
    joined = "\n\n".join(blocks)
    if len(joined) > config.SYNTHESIS_MAX_INPUT_CHARS:
        joined = joined[: config.SYNTHESIS_MAX_INPUT_CHARS] + "\n[...]"

    prompt = f"""
Tu es un arbitre de synthese. A partir des reponses suivantes, produis
une reponse finale claire, exacte et coherente, en resolvant les contradictions.
Si une incertitude subsiste, indique-la explicitement.

Question utilisateur:
{question}

Reponses candidates:
{joined}
"""
    return prompt
