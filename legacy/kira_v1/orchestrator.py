import time

import requests
try:
    import ollama
except Exception:
    ollama = None

import config
from audit import log


def _get_perf_setting(key: str, fallback):
    try:
        settings = config.get_perf_settings()
        return settings.get(key, fallback)
    except Exception:
        return fallback


_GROQ_CIRCUIT = {
    "disabled_until": 0.0,
    "last_error": "",
    "trips": 0,
}


def _circuit_open():
    if not config.GROQ_CIRCUIT_BREAKER_ENABLED:
        return False
    return time.time() < _GROQ_CIRCUIT["disabled_until"]


def _trip_circuit(reason: str):
    if not config.GROQ_CIRCUIT_BREAKER_ENABLED:
        return
    cooldown = max(0, int(config.GROQ_CIRCUIT_BREAKER_SECONDS))
    _GROQ_CIRCUIT["disabled_until"] = time.time() + cooldown
    _GROQ_CIRCUIT["last_error"] = reason
    _GROQ_CIRCUIT["trips"] += 1
    log(
        "cloud_circuit_open",
        {
            "cooldown_seconds": cooldown,
            "disabled_until": _GROQ_CIRCUIT["disabled_until"],
            "reason": reason,
            "trips": _GROQ_CIRCUIT["trips"],
        },
    )


def query_groq(prompt: str):
    if not config.USE_CLOUD or not config.GROQ_API_KEY:
        return None

    if _circuit_open():
        remaining = int(_GROQ_CIRCUIT["disabled_until"] - time.time())
        log(
            "cloud_circuit_skip",
            {
                "remaining_seconds": max(0, remaining),
                "last_error": _GROQ_CIRCUIT["last_error"],
            },
        )
        return None

    max_tokens = _get_perf_setting("groq_max_tokens", config.GROQ_MAX_TOKENS)
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": config.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "Tu es un assistant rapide et précis."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    last_error = None
    retries = 0 if config.GROQ_DISABLE_RETRIES else config.GROQ_MAX_RETRIES
    for attempt in range(retries + 1):
        start = time.monotonic()
        try:
            response = None
            response = requests.post(
                url, headers=headers, json=data, timeout=config.GROQ_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            log(
                "cloud_query_success",
                {
                    "provider": "groq",
                    "model": config.GROQ_MODEL,
                    "prompt_length": len(prompt),
                    "max_tokens": max_tokens,
                    "usage_tokens": response.json().get("usage", {}),
                    "duration_ms": int((time.monotonic() - start) * 1000),
                },
            )
            return content
        except requests.HTTPError as e:
            status = response.status_code if response is not None else None
            body = ""
            if response is not None:
                try:
                    body = response.text.strip()
                except Exception:
                    body = ""
            last_error = f"{status} {body}".strip() if status else str(e)
            if status == 429:
                _trip_circuit(last_error)
                log(
                    "cloud_query_rate_limited",
                    {"status_code": status, "response_body": body[:500]},
                )
                break
            log(
                "cloud_query_error",
                {
                    "error": str(e),
                    "status_code": status,
                    "response_body": body[:500],
                    "attempt": attempt + 1,
                },
            )
            if attempt < retries:
                time.sleep(config.GROQ_RETRY_BACKOFF_SECONDS * (attempt + 1))
        except Exception as e:
            last_error = e
            log(
                "cloud_query_error",
                {
                    "error": str(e),
                    "attempt": attempt + 1,
                },
            )
            if attempt < retries:
                time.sleep(config.GROQ_RETRY_BACKOFF_SECONDS * (attempt + 1))

    return f"[Erreur Groq] {str(last_error)}"


def query_models(prompt: str, models=None):
    if models is None:
        models = config.DEFAULT_MODELS

    if getattr(config, "USE_LITE", False) and not (config.USE_CLOUD or config.USE_LOCAL):
        from lite import respond as lite_respond
        start = time.monotonic()
        response = lite_respond(prompt)
        log(
            "lite_response",
            {
                "prompt_length": len(prompt),
                "duration_ms": int((time.monotonic() - start) * 1000),
            },
        )
        return response

    responses = {}

    # 1. Priorité Groq si activé
    cloud_ok = False
    if config.USE_CLOUD:
        groq_response = query_groq(prompt)
        if groq_response:
            responses["groq"] = groq_response
            if not groq_response.startswith("[Erreur"):
                cloud_ok = True

    # 2. Fallback local (offline)
    should_try_local = config.USE_LOCAL and (
        (not config.LOCAL_FALLBACK_ONLY) or (not cloud_ok)
    )
    if should_try_local:
        if ollama is None:
            log("local_query_error", {"model": "ollama", "error": "ollama_not_available"})
            should_try_local = False
    if should_try_local:
        local_num_predict = _get_perf_setting(
            "local_num_predict", config.LOCAL_NUM_PREDICT
        )
        for model in models:
            try:
                start = time.monotonic()
                res = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"num_predict": local_num_predict},
                )
                responses[model] = res["message"]["content"]
                log(
                    "local_query",
                    {
                        "model": model,
                        "prompt_length": len(prompt),
                        "duration_ms": int((time.monotonic() - start) * 1000),
                        "num_predict": local_num_predict,
                    },
                )
            except Exception as e:
                responses[model] = f"[Erreur locale] {str(e)}"
                log(
                    "local_query_error",
                    {
                        "model": model,
                        "error": str(e),
                        "num_predict": local_num_predict,
                    },
                )

    # Fallback lite si rien
    if not responses and getattr(config, "USE_LITE", False):
        from lite import respond as lite_respond
        start = time.monotonic()
        response = lite_respond(prompt)
        log(
            "lite_response",
            {
                "prompt_length": len(prompt),
                "duration_ms": int((time.monotonic() - start) * 1000),
            },
        )
        return response

    # Si aucune réponse, on le signale
    if not responses:
        return "Aucune réponse disponible (ni cloud ni local)."

    return responses
