"""API HTTP de KIRA (Starlette). Tout est protégé par le jeton du propriétaire, sauf /api/auth/login."""
from __future__ import annotations

import hashlib
import json
import queue
import threading
from pathlib import Path
from typing import Callable

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import VERSION, agent, audit, auth, briefing, budget, config, consolidate, devices, evolution, files, jobs, memory, veille
from .db import get_db, jload, now_iso
from .llm import get_router
from .tools import specs as tool_specs


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Ctx:
    def __init__(self, request: Request, body: dict):
        self.request = request
        self.body = body
        self.path = request.path_params
        self.query = dict(request.query_params)
        self.device: dict | None = None  # renseigné pour les routes de la machine (access="device")

    def int_query(self, name: str, default: int, low: int = 0, high: int = 1000) -> int:
        try:
            return max(low, min(high, int(self.query.get(name, default))))
        except (TypeError, ValueError):
            return default

    def text(self, name: str, limit: int = 20000) -> str:
        value = self.body.get(name, "")
        if not isinstance(value, str):
            raise ApiError(400, f"« {name} » doit être du texte.")
        if len(value) > limit:
            raise ApiError(400, f"« {name} » est trop long ({limit} caractères maximum).")
        return value

    def id(self, name: str = "id") -> int:
        try:
            return int(self.path[name])
        except (KeyError, ValueError):
            raise ApiError(404, "Introuvable.")


# -- accès ---------------------------------------------------------------
def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()  # ajouté par le proxy de l'hébergeur
    return request.client.host if request.client else "inconnu"


def _is_owner(request: Request) -> bool:
    header = request.headers.get("authorization", "")
    return header.lower().startswith("bearer ") and auth.verify_token(header[7:].strip()) is not None


def _require_owner(request: Request) -> None:
    if not _is_owner(request):
        raise ApiError(401, "Connexion requise ou session expirée.")


def _require_cron(request: Request) -> None:
    if _is_owner(request) or auth.check_cron_token(request.headers.get("x-cron-token", "")):
        return
    raise ApiError(401, "Jeton de tâche planifiée invalide.")


def _require_device(request: Request) -> dict:
    """Routes réservées à une machine appairée : ``Authorization: Device kdev_…`` (le jeton du propriétaire n'y ouvre rien)."""
    device = devices.authenticate(request.headers.get("authorization", ""))
    if not device:
        raise ApiError(401, "Appareil inconnu ou révoqué.")
    return device


BODY_LIMIT = 1_000_000          # routes du propriétaire
DEVICE_BODY_LIMIT = 64_000      # appairage et scrutation
RESULT_BODY_LIMIT = 18_000_000  # résultat d'une machine (jusqu'à 3 captures en base 64)


async def _read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise ApiError(413, "Requête trop volumineuse.")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise ApiError(413, "Requête trop volumineuse.")
        chunks.append(chunk)
    return b"".join(chunks)


def endpoint(fn: Callable[[Ctx], object], access: str = "owner", body_limit: int | None = None):
    async def handler(request: Request):
        try:
            device = None
            if access == "owner":
                _require_owner(request)
            elif access == "cron":
                _require_cron(request)
            elif access == "device":
                device = await run_in_threadpool(_require_device, request)  # lit la base : hors de la boucle d'événements
            body: dict = {}
            if request.method in ("POST", "PUT", "PATCH"):
                raw = await _read_body(request, body_limit or (DEVICE_BODY_LIMIT if access == "device" else BODY_LIMIT))
                if raw:
                    try:
                        body = json.loads(raw)
                    except ValueError:
                        raise ApiError(400, "JSON invalide.")
                    if not isinstance(body, dict):
                        raise ApiError(400, "Un objet JSON est attendu.")
            ctx = Ctx(request, body)
            ctx.device = device
            result = await run_in_threadpool(fn, ctx)
            return result if isinstance(result, Response) else JSONResponse(result)
        except ApiError as exc:
            return JSONResponse({"error": exc.message}, status_code=exc.status)
        except (evolution.EvolutionError, devices.DeviceError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # noqa: BLE001
            audit.log("api_error", {"path": request.url.path, "error": f"{type(exc).__name__}: {exc}"[:300]})
            return JSONResponse({"error": "Erreur interne du serveur."}, status_code=500)

    return handler


_global_throttle = auth.LoginThrottle(max_failures=25, window=300)


# -- authentification ----------------------------------------------------
def h_login(ctx: Ctx):
    ip = client_ip(ctx.request)
    if auth.throttle.blocked(ip) or _global_throttle.blocked("*"):
        raise ApiError(429, "Trop d'essais. Réessayez dans quelques minutes.")
    if not config.settings.owner_password:
        raise ApiError(503, "OWNER_PASSWORD n'est pas configuré sur le serveur.")
    if not auth.check_password(str(ctx.body.get("password", ""))):
        auth.throttle.fail(ip)
        _global_throttle.fail("*")
        audit.log("login_failed", {"ip": ip})
        raise ApiError(401, "Mot de passe incorrect.")
    auth.throttle.ok(ip)
    audit.log("login", {"ip": ip}, actor="owner")
    return {"token": auth.make_token(), "name": config.settings.owner_name}


def h_me(ctx: Ctx):
    return {"name": config.settings.owner_name, "version": VERSION}


# -- état ----------------------------------------------------------------
def _devices_summary() -> dict:
    rows = devices.list_devices()
    return {"total": len(rows), "online": sum(1 for d in rows if d["online"]), "pending": len(devices.pending_actions()),
            "paused": devices.global_paused()}


def h_status(ctx: Ctx):
    db = get_db()
    s = config.settings
    return {
        "version": VERSION,
        "owner": s.owner_name,
        "database": db.kind,
        "providers": get_router().status(),
        "budget": budget.summary(),
        "tools": [t.name for t in tool_specs()],
        "memory": memory.counts(),
        "veille": {**veille.counts(), "last_run": db.kv_get("veille_last_run") or None},
        "proposals_draft": int((db.q1("SELECT COUNT(*) AS n FROM proposals WHERE status = 'draft'") or {"n": 0})["n"]),
        "github": bool(s.github_token),
        "web_search": "tavily" if s.tavily_api_key else "duckduckgo",
        "jobs": jobs.status(),
        "devices": _devices_summary(),
    }


# -- conversations -------------------------------------------------------
def h_conversations(ctx: Ctx):
    return {
        "conversations": get_db().q(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
            [ctx.int_query("limit", 100, 1, 300)],
        )
    }


def _conversation_or_404(cid: str) -> dict:
    conv = get_db().q1("SELECT * FROM conversations WHERE id = ?", [cid])
    if not conv:
        raise ApiError(404, "Conversation introuvable.")
    return conv


def h_conversation_messages(ctx: Ctx):
    conv = _conversation_or_404(ctx.path["id"])
    rows = get_db().q(
        "SELECT id, role, content, meta, feedback, created_at FROM messages WHERE conversation_id = ? ORDER BY id LIMIT 600",
        [conv["id"]],
    )
    for r in rows:
        r["meta"] = jload(r["meta"], {})
    return {"conversation": conv, "messages": rows}


def h_conversation_patch(ctx: Ctx):
    conv = _conversation_or_404(ctx.path["id"])
    title = ctx.text("title", 120).strip()
    if not title:
        raise ApiError(400, "Le titre est vide.")
    get_db().update("conversations", conv["id"], {"title": title})
    return {"ok": True}


def h_conversation_delete(ctx: Ctx):
    conv = _conversation_or_404(ctx.path["id"])
    get_db().run("DELETE FROM messages WHERE conversation_id = ?", [conv["id"]])
    get_db().run("DELETE FROM conversations WHERE id = ?", [conv["id"]])
    audit.log("conversation_deleted", {"id": conv["id"]}, actor="owner")
    return {"ok": True}


def h_chat(ctx: Ctx):
    message = ctx.text("message")
    tier = ctx.body.get("tier", "default")
    if tier not in ("default", "deep"):
        tier = "default"
    conversation_id = ctx.body.get("conversation_id") or None

    # Le tour tourne dans son propre thread : si le téléphone coupe la connexion (écran verrouillé, tunnel),
    # la réponse est quand même terminée et enregistrée. Le flux ne fait que la relayer.
    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            for event in agent.run_turn(conversation_id, message, tier):
                events.put(event)
        except Exception as exc:  # noqa: BLE001
            audit.log("chat_stream_error", {"error": f"{type(exc).__name__}: {exc}"[:300]})
            events.put({"type": "error", "message": "Erreur interne."})
        finally:
            events.put(None)

    threading.Thread(target=work, name="chat-turn", daemon=True).start()

    def stream():
        while True:
            try:
                event = events.get(timeout=15)
            except queue.Empty:
                yield ": ping\n\n"  # garde la connexion ouverte à travers les proxys pendant une longue réflexion
                continue
            if event is None:
                break
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        yield 'data: {"type": "end"}\n\n'

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


def h_feedback(ctx: Ctx):
    mid = ctx.id()
    value = ctx.body.get("value")
    if value not in (-1, 0, 1):
        raise ApiError(400, "La valeur doit être 1, 0 ou -1.")
    row = get_db().q1("SELECT id FROM messages WHERE id = ? AND role = 'assistant'", [mid])
    if not row:
        raise ApiError(404, "Message introuvable.")
    note = ctx.text("note", 500) if "note" in ctx.body else None
    values = {"feedback": None if value == 0 else value}
    if note is not None:
        values["feedback_note"] = note
    get_db().update("messages", mid, values)
    audit.log("feedback", {"message": mid, "value": value}, actor="owner")
    return {"ok": True}


# -- mémoire -------------------------------------------------------------
def h_memory_list(ctx: Ctx):
    q = ctx.query.get("q", "").strip()
    kind = ctx.query.get("kind", "")
    limit = ctx.int_query("limit", 100, 1, 300)
    status = ctx.query.get("status", "active")
    if status not in ("active", "archived", "superseded", ""):
        raise ApiError(400, "État de souvenir invalide.")
    if q:
        items = memory.search(q, kinds=(kind,) if kind in memory.KINDS else None, limit=limit, status=status)
    else:
        items = memory.list_items(kind if kind in memory.KINDS else "", limit, ctx.int_query("offset", 0), status=status)
    return {"items": items, "counts": memory.counts(status), "semantic": memory.semantic_status()}


def h_memory_create(ctx: Ctx):
    item = memory.add(ctx.body.get("kind", "note"), ctx.text("content", 4000), str(ctx.body.get("tags", "")), source="manuel",
                      importance=ctx.body.get("importance", 0.5), confidence=ctx.body.get("confidence", 1.0))
    audit.log("memory_added", {"id": item["id"]}, actor="owner")
    return item


def h_memory_patch(ctx: Ctx):
    mid = ctx.id()
    if not memory.get(mid):
        raise ApiError(404, "Souvenir introuvable.")
    item = memory.update(
        mid,
        content=ctx.text("content", 4000) if "content" in ctx.body else None,
        kind=ctx.body.get("kind"),
        tags=str(ctx.body["tags"]) if "tags" in ctx.body else None,
        importance=ctx.body.get("importance"), confidence=ctx.body.get("confidence"), status=ctx.body.get("status"),
    )
    audit.log("memory_updated", {"id": mid}, actor="owner")
    return item


def h_memory_delete(ctx: Ctx):
    mid = ctx.id()
    if not memory.delete(mid):
        raise ApiError(404, "Souvenir introuvable.")
    audit.log("memory_deleted", {"id": mid}, actor="owner")
    return {"ok": True}


def h_memory_history(ctx: Ctx):
    mid = ctx.id()
    if not memory.get(mid):
        raise ApiError(404, "Souvenir introuvable.")
    return {"items": memory.history(mid)}


def h_memory_supersede(ctx: Ctx):
    mid = ctx.id()
    if not memory.get(mid):
        raise ApiError(404, "Souvenir introuvable.")
    return memory.supersede(mid, ctx.text("content", 4000), kind=ctx.body.get("kind"))


def h_identity(ctx: Ctx):
    return {"identity": memory.identity(), "items": memory.list_items("identity", limit=40)}


# -- veille --------------------------------------------------------------
def h_veille_items(ctx: Ctx):
    status = ctx.query.get("status", "pending")
    if status not in ("pending", "validated", "rejected"):
        raise ApiError(400, "Statut inconnu.")
    return {"items": veille.list_items(status, ctx.int_query("limit", 60, 1, 200)), "counts": veille.counts()}


def h_veille_validate(ctx: Ctx):
    item = veille.validate(ctx.id())
    if not item:
        raise ApiError(404, "Élément introuvable.")
    return item


def h_veille_reject(ctx: Ctx):
    item = veille.reject(ctx.id())
    if not item:
        raise ApiError(404, "Élément introuvable.")
    return item


def h_veille_validate_all(ctx: Ctx):
    try:
        min_score = float(ctx.body.get("min_score", 0.7))
    except (TypeError, ValueError):
        raise ApiError(400, "min_score doit être un nombre.")
    return {"validated": veille.validate_many(min_score)}


def h_veille_sources(ctx: Ctx):
    return {"sources": veille.list_sources()}


def h_veille_source_add(ctx: Ctx):
    return veille.add_source(ctx.text("url", 500), ctx.text("name", 120) if "name" in ctx.body else "",
                             ctx.body.get("kind", "rss"), ctx.body.get("score", 0.5))


def h_veille_source_patch(ctx: Ctx):
    row = veille.update_source(
        ctx.id(), name=ctx.body.get("name"), score=ctx.body.get("score"), active=ctx.body.get("active"), kind=ctx.body.get("kind")
    )
    if not row:
        raise ApiError(404, "Source introuvable.")
    return row


def h_veille_source_delete(ctx: Ctx):
    if not veille.delete_source(ctx.id()):
        raise ApiError(404, "Source introuvable.")
    return {"ok": True}


def h_veille_run(ctx: Ctx):
    started = jobs.start("veille", veille.run)
    return JSONResponse({"started": started, "message": "Veille lancée." if started else "Une veille est déjà en cours."},
                        status_code=202)


# -- évolution -----------------------------------------------------------
def h_evo_list(ctx: Ctx):
    return {"proposals": evolution.list_proposals(), "jobs": jobs.status()}


def h_evo_propose(ctx: Ctx):
    goal = ctx.text("goal", 1500) if "goal" in ctx.body else ""
    started = jobs.start("evolution", lambda: evolution.propose(goal, trigger="manual"))
    return JSONResponse({"started": started, "message": "Analyse lancée (1 à 3 minutes)." if started else
                         "Une analyse est déjà en cours."}, status_code=202)


def h_evo_approve(ctx: Ctx):
    return evolution.approve(ctx.id())


def h_evo_reject(ctx: Ctx):
    result = evolution.reject(ctx.id())
    if not result:
        raise ApiError(404, "Proposition introuvable.")
    return result


# -- appareils : le « cœur » ----------------------------------------------
CORE_AGENT = Path(__file__).resolve().parents[1] / "core" / "kira_core.py"
_pair_global = auth.LoginThrottle(max_failures=40, window=600)


def _device_id(ctx: Ctx) -> int:
    row = devices.get_device(ctx.id())
    if not row:
        raise ApiError(404, "Appareil introuvable.")
    return row["id"]


def h_devices(ctx: Ctx):
    return {
        "devices": devices.list_devices(),
        "paused": devices.global_paused(),
        "categories": devices.CATEGORIES,
        "never_auto": sorted(devices.NEVER_AUTO),
        "pending": len(devices.pending_actions()),
    }


def h_device_pair_code(ctx: Ctx):
    result = devices.create_pair_code()
    if CORE_AGENT.exists():
        result["agent_sha256"] = hashlib.sha256(CORE_AGENT.read_bytes()).hexdigest()
    return result


def h_device_patch(ctx: Ctx):
    did = _device_id(ctx)
    out = None
    if "name" in ctx.body:
        out = devices.rename(did, ctx.text("name", 80))
    if "policy" in ctx.body:
        if not isinstance(ctx.body["policy"], dict):
            raise ApiError(400, "« policy » doit être un objet.")
        out = devices.set_policy(did, ctx.body["policy"])
    if "paused" in ctx.body:
        out = devices.set_paused(did, bool(ctx.body["paused"]))
    if out is None:
        raise ApiError(400, "Rien à modifier.")
    return out


def h_device_revoke(ctx: Ctx):
    devices.revoke(_device_id(ctx))
    return {"ok": True}


def h_device_actions(ctx: Ctx):
    return {"actions": devices.list_actions(_device_id(ctx), ctx.int_query("limit", 30, 1, 100))}


def h_actions_pending(ctx: Ctx):
    return {"actions": devices.pending_actions(), "paused": devices.global_paused()}


def h_actions_recent(ctx: Ctx):
    return {"actions": devices.list_actions(None, ctx.int_query("limit", 40, 1, 150))}


def h_action_get(ctx: Ctx):
    action = devices.get_action(ctx.id())
    if not action:
        raise ApiError(404, "Demande introuvable.")
    return action


def h_action_approve(ctx: Ctx):
    return devices.approve(ctx.id())


def h_action_deny(ctx: Ctx):
    return devices.deny(ctx.id())


def h_devices_pause_all(ctx: Ctx):
    devices.set_global_pause(bool(ctx.body.get("paused", True)))
    return {"paused": devices.global_paused()}


# côté machine -----------------------------------------------------------
def _info(ctx: Ctx) -> dict:
    info = ctx.body.get("info", {})
    if not isinstance(info, dict):
        raise ApiError(400, "« info » doit être un objet.")
    return info


def h_core_pair(ctx: Ctx):
    ip = client_ip(ctx.request)
    if devices.pair_throttle.blocked(ip) or _pair_global.blocked("*"):
        raise ApiError(429, "Trop d'essais d'appairage. Réessaie dans quelques minutes.")
    try:
        result = devices.pair(ctx.text("code", 40), ctx.text("name", 80), ctx.text("platform", 30), _info(ctx),
                              ctx.text("version", 20))
    except devices.DeviceError:
        devices.pair_throttle.fail(ip)
        _pair_global.fail("*")
        audit.log("device_pair_failed", {"ip": ip})
        raise
    devices.pair_throttle.ok(ip)
    return result


def h_core_poll(ctx: Ctx):
    try:
        wait = float(ctx.body.get("wait", 20))
    except (TypeError, ValueError):
        wait = 20.0
    return devices.poll(ctx.device, _info(ctx), max(0.0, min(25.0, wait)))


def h_core_ping(ctx: Ctx):
    return devices.touch(ctx.device)


def h_core_result(ctx: Ctx):
    try:
        action_id = int(ctx.body.get("action_id"))
    except (TypeError, ValueError):
        raise ApiError(400, "« action_id » invalide.")
    images = ctx.body.get("images") or []
    if not isinstance(images, list):
        raise ApiError(400, "« images » doit être une liste.")
    done = devices.submit_result(ctx.device, action_id, bool(ctx.body.get("ok")), str(ctx.body.get("output", "")), images)
    return {"ok": True, "status": done["status"]}


def h_core_agent(ctx: Ctx):
    if not CORE_AGENT.exists():
        raise ApiError(404, "Le programme du cœur n'est pas installé sur ce serveur.")
    return Response(CORE_AGENT.read_bytes(), media_type="text/x-python; charset=utf-8",
                    headers={"Cache-Control": "no-cache", "Content-Disposition": 'attachment; filename="kira_core.py"'})


# -- divers --------------------------------------------------------------
def h_audit(ctx: Ctx):
    return {"entries": audit.recent(ctx.int_query("limit", 80, 1, 300), ctx.query.get("prefix", ""))}


def h_file(ctx: Ctx):
    row = files.get(ctx.path["id"])
    if not row:
        raise ApiError(404, "Fichier introuvable.")
    # captures et graphiques : jamais gardés dans le cache du navigateur
    return Response(row["data"], media_type=row["mime"], headers={"Cache-Control": "private, no-store"})


def h_export(ctx: Ctx):
    db = get_db()
    dump = {
        "exported_at": now_iso(),
        "version": VERSION,
        "conversations": db.q("SELECT * FROM conversations ORDER BY created_at"),
        "messages": db.q("SELECT id, conversation_id, role, content, feedback, created_at FROM messages ORDER BY id"),
        "memories": db.q("SELECT * FROM memories ORDER BY id"),
        "memory_history": db.q("SELECT * FROM memory_history ORDER BY memory_id, version"),
        "identity": memory.identity(),
        "veille_sources": db.q("SELECT * FROM veille_sources ORDER BY id"),
        "veille_items": db.q("SELECT * FROM veille_items ORDER BY id"),
        "proposals": db.q("SELECT id, title, rationale, diff, status, pr_url, created_at FROM proposals ORDER BY id"),
    }
    audit.log("export", {}, actor="owner")
    return Response(
        json.dumps(dump, ensure_ascii=False, indent=1),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="kira-export.json"'},
    )


def _daily() -> dict:
    result: dict = {"veille": veille.run()}
    try:
        result["appareils"] = devices.cleanup()
    except Exception as exc:  # noqa: BLE001
        result["appareils"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    try:
        result["consolidation"] = consolidate.consolidate()
    except Exception as exc:  # noqa: BLE001
        result["consolidation"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    if memory.semantic_status()["configured"]:
        result["memory_index"] = memory.backfill(20)
    return result


def h_cron_daily(ctx: Ctx):
    started = jobs.start("daily", _daily)
    return JSONResponse({"started": started}, status_code=202)


def h_briefing(ctx: Ctx):
    return briefing.build()


def h_health(ctx: Ctx):
    return {"ok": True, "version": VERSION}


def routes() -> list[Route]:
    def r(path: str, fn, methods: list[str], access: str = "owner") -> Route:
        return Route(path, endpoint(fn, access), methods=methods)

    return [
        r("/healthz", h_health, ["GET"], "none"),
        r("/api/auth/login", h_login, ["POST"], "none"),
        r("/api/auth/me", h_me, ["GET"]),
        r("/api/status", h_status, ["GET"]),
        r("/api/briefing", h_briefing, ["GET"]),
        r("/api/conversations", h_conversations, ["GET"]),
        r("/api/conversations/{id}", h_conversation_messages, ["GET"]),
        r("/api/conversations/{id}", h_conversation_patch, ["PATCH"]),
        r("/api/conversations/{id}", h_conversation_delete, ["DELETE"]),
        r("/api/chat", h_chat, ["POST"]),
        r("/api/messages/{id:int}/feedback", h_feedback, ["POST"]),
        r("/api/memory", h_memory_list, ["GET"]),
        r("/api/memory", h_memory_create, ["POST"]),
        r("/api/identity", h_identity, ["GET"]),
        r("/api/memory/{id:int}/history", h_memory_history, ["GET"]),
        r("/api/memory/{id:int}/supersede", h_memory_supersede, ["POST"]),
        r("/api/memory/{id:int}", h_memory_patch, ["PATCH"]),
        r("/api/memory/{id:int}", h_memory_delete, ["DELETE"]),
        r("/api/veille/items", h_veille_items, ["GET"]),
        r("/api/veille/items/{id:int}/validate", h_veille_validate, ["POST"]),
        r("/api/veille/items/{id:int}/reject", h_veille_reject, ["POST"]),
        r("/api/veille/validate-all", h_veille_validate_all, ["POST"]),
        r("/api/veille/sources", h_veille_sources, ["GET"]),
        r("/api/veille/sources", h_veille_source_add, ["POST"]),
        r("/api/veille/sources/{id:int}", h_veille_source_patch, ["PATCH"]),
        r("/api/veille/sources/{id:int}", h_veille_source_delete, ["DELETE"]),
        r("/api/veille/run", h_veille_run, ["POST"]),
        r("/api/evolution", h_evo_list, ["GET"]),
        r("/api/evolution/propose", h_evo_propose, ["POST"]),
        r("/api/evolution/{id:int}/approve", h_evo_approve, ["POST"]),
        r("/api/evolution/{id:int}/reject", h_evo_reject, ["POST"]),
        r("/api/devices", h_devices, ["GET"]),
        r("/api/devices/pair-code", h_device_pair_code, ["POST"]),
        r("/api/devices/pause-all", h_devices_pause_all, ["POST"]),
        r("/api/devices/{id:int}", h_device_patch, ["PATCH"]),
        r("/api/devices/{id:int}", h_device_revoke, ["DELETE"]),
        r("/api/devices/{id:int}/actions", h_device_actions, ["GET"]),
        r("/api/actions", h_actions_recent, ["GET"]),
        r("/api/actions/pending", h_actions_pending, ["GET"]),
        r("/api/actions/{id:int}", h_action_get, ["GET"]),
        r("/api/actions/{id:int}/approve", h_action_approve, ["POST"]),
        r("/api/actions/{id:int}/deny", h_action_deny, ["POST"]),
        Route("/api/core/pair", endpoint(h_core_pair, "none", DEVICE_BODY_LIMIT), methods=["POST"]),
        Route("/api/core/poll", endpoint(h_core_poll, "device"), methods=["POST"]),
        Route("/api/core/ping", endpoint(h_core_ping, "device"), methods=["POST"]),
        Route("/api/core/result", endpoint(h_core_result, "device", RESULT_BODY_LIMIT), methods=["POST"]),
        r("/core/kira_core.py", h_core_agent, ["GET"], "none"),
        r("/api/audit", h_audit, ["GET"]),
        r("/api/files/{id}", h_file, ["GET"]),
        r("/api/export", h_export, ["GET"]),
        r("/api/cron/daily", h_cron_daily, ["POST"], "cron"),
    ]
