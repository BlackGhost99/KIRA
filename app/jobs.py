"""Tâches de fond (veille, consolidation) : lancées en arrière-plan, une seule à la fois par nom."""
from __future__ import annotations

import threading
from typing import Callable

from . import audit
from .db import now_iso

_lock = threading.Lock()
_running: dict[str, str] = {}  # nom -> heure de début
_last: dict[str, dict] = {}


def start(name: str, fn: Callable[[], dict], wait: bool = False) -> bool:
    """Lance ``fn`` en arrière-plan. Renvoie False si la tâche tourne déjà."""
    with _lock:
        if name in _running:
            return False
        _running[name] = now_iso()

    def work() -> None:
        started = _running.get(name, "")
        try:
            result = fn()
            _last[name] = {"ok": True, "started": started, "finished": now_iso(), "result": result}
        except Exception as exc:  # noqa: BLE001
            audit.log("job_error", {"job": name, "error": f"{type(exc).__name__}: {exc}"[:300]})
            _last[name] = {"ok": False, "started": started, "finished": now_iso(), "error": str(exc)[:300]}
        finally:
            with _lock:
                _running.pop(name, None)

    thread = threading.Thread(target=work, name=f"job-{name}", daemon=True)
    thread.start()
    if wait:
        thread.join()
    return True


def status() -> dict:
    with _lock:
        return {"running": dict(_running), "last": dict(_last)}
