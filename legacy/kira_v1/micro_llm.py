import json
import subprocess
import time
from urllib import request

import config
from audit import log


def _post_json(url: str, payload: dict, timeout: int):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"text": raw}


def _extract_text(response: dict | str):
    if isinstance(response, str):
        return response.strip()
    for key in ["text", "response", "content", "output"]:
        if key in response and isinstance(response[key], str):
            return response[key].strip()
    if "choices" in response and isinstance(response["choices"], list):
        choice = response["choices"][0]
        if isinstance(choice, dict):
            if "text" in choice:
                return str(choice["text"]).strip()
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content")
                if isinstance(content, str):
                    return content.strip()
    return ""


def _run_command(command: str, prompt: str, timeout: int):
    try:
        result = subprocess.run(
            command,
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            shell=True,
        )
        output = result.stdout.decode("utf-8", errors="ignore").strip()
        return output
    except Exception as e:
        log("micro_llm_error", {"mode": "command", "error": str(e)})
        return ""


def generate(prompt: str):
    if not config.MICRO_LLM_ENABLED:
        return ""
    timeout = int(getattr(config, "MICRO_LLM_TIMEOUT_SECONDS", 3))
    mode = getattr(config, "MICRO_LLM_MODE", "endpoint")
    start = time.monotonic()
    text = ""

    if mode == "command":
        cmd = (config.MICRO_LLM_COMMAND or "").strip()
        if not cmd:
            return ""
        text = _run_command(cmd, prompt, timeout)
    else:
        endpoint = (config.MICRO_LLM_ENDPOINT or "").strip()
        if not endpoint:
            return ""
        payload = {
            "prompt": prompt,
            "temperature": getattr(config, "MICRO_LLM_TEMPERATURE", 0.4),
        }
        try:
            resp = _post_json(endpoint, payload, timeout)
            text = _extract_text(resp)
        except Exception as e:
            log("micro_llm_error", {"mode": "endpoint", "error": str(e)})
            return ""

    max_chars = getattr(config, "MICRO_LLM_MAX_OUTPUT_CHARS", 1800)
    if isinstance(max_chars, int) and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    log(
        "micro_llm_response",
        {
            "mode": mode,
            "prompt_length": len(prompt),
            "duration_ms": int((time.monotonic() - start) * 1000),
            "chars": len(text),
        },
    )
    return text
