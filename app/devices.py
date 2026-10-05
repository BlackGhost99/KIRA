"""Le « cœur » côté serveur : appareils de Brice, appairage, politique par niveaux et demandes d'accord.

Principe : le cœur installé sur une machine ne reçoit JAMAIS d'ordre qui ne soit pas passé par ``request_action`` ci-dessous.
Chaque action est classée dans une catégorie, et chaque catégorie a un mode choisi par Brice :
  auto = libre · ask = Brice valide dans l'appli · deny = interdit.
La machine applique en plus ses propres plafonds locaux (catégories activées, dossiers autorisés) que le serveur ne peut
pas élargir à distance. Tout est journalisé.

FICHIER PROTÉGÉ : l'évolution ne peut pas le modifier.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from datetime import datetime, timedelta, timezone

from . import audit, files
from .auth import LoginThrottle
from .db import get_db, jdump, jload, now_iso

# ---------------------------------------------------------------- catégories et politique
CATEGORIES = {
    "monitor": "Surveiller l'état de la machine",
    "fs_read": "Lire des fichiers",
    "fs_write": "Écrire des fichiers",
    "fs_delete": "Supprimer des fichiers",
    "exec": "Lancer des commandes",
    "screen": "Voir l'écran",
    "input": "Piloter souris et clavier",
}
MODES = ("auto", "ask", "deny")
DEFAULT_POLICY = {
    "monitor": "auto",
    "fs_read": "auto",
    "fs_write": "ask",
    "fs_delete": "ask",
    "exec": "ask",
    "screen": "ask",
    "input": "ask",
}
NEVER_AUTO = frozenset({"fs_delete"})  # la suppression demande toujours l'accord de Brice

ONLINE_WINDOW_S = 75
PAIR_TTL_S = 600
PENDING_TTL_S = 600
QUEUED_TTL_S = 300
RUNNING_TTL_S = 900
MAX_RESULT_CHARS = 60_000
MAX_IMAGE_BYTES = 4_000_000  # par image ; l'agent réduit déjà ses captures (≈ 0,3 Mo)
TERMINAL = ("done", "failed", "denied", "expired", "cancelled")

pair_throttle = LoginThrottle(max_failures=8, window=600)


class DeviceError(Exception):
    """Erreur affichable telle quelle à Brice."""


# ---------------------------------------------------------------- catalogue des actions
# kind -> (catégorie, risque, {argument: spécification})
#   ("str", longueur_max, obligatoire) · ("int", min, max, défaut) · ("bool", défaut) · ("enum", choix, défaut)
_PATH = ("str", 2000, True)
# Ce que Brice approuve doit être exactement ce qui s'exécute : on n'accepte donc que ce qu'on peut lui montrer EN ENTIER
# (pas de début anodin suivi d'une suite cachée). Pour un fichier plus long : plusieurs écritures en mode « append »,
# chacune montrée en entier.
MAX_SHOWN_WRITE = 20_000
MAX_SHOWN_CODE = 6_000
KINDS: dict[str, tuple[str, str, dict]] = {
    "status": ("monitor", "low", {}),
    "processes": ("monitor", "low", {"limit": ("int", 1, 100, 25)}),
    "list_dir": ("fs_read", "low", {"path": _PATH}),
    "read_file": ("fs_read", "low", {"path": _PATH, "max_bytes": ("int", 1, 1_000_000, 100_000),
                                      "offset": ("int", 0, 2_000_000_000, 0)}),
    "search_files": ("fs_read", "low", {"path": _PATH, "query": ("str", 200, True), "max_results": ("int", 1, 200, 50),
                                         "contents": ("bool", False)}),
    "write_file": ("fs_write", "medium", {"path": _PATH, "content": ("str", MAX_SHOWN_WRITE, True),
                                           "mode": ("enum", ("create", "overwrite", "append"), "create")}),
    "make_dir": ("fs_write", "medium", {"path": _PATH}),
    "move": ("fs_write", "medium", {"src": _PATH, "dst": _PATH}),
    "delete": ("fs_delete", "high", {"path": _PATH}),
    "run_command": ("exec", "high", {"command": ("str", 4000, True), "cwd": ("str", 2000, False),
                                      "timeout": ("int", 1, 600, 60)}),
    "run_python": ("exec", "high", {"code": ("str", MAX_SHOWN_CODE, True), "timeout": ("int", 1, 600, 60)}),
    "open": ("exec", "medium", {"target": ("str", 2000, True)}),
    "screenshot": ("screen", "medium", {}),
    "mouse": ("input", "medium", {"action": ("enum", ("move", "click", "double_click", "right_click", "scroll"), "click"),
                                   "x": ("int", 0, 20000, None), "y": ("int", 0, 20000, None),
                                   "amount": ("int", -50, 50, 0), "space": ("enum", ("screenshot", "screen"), "screenshot")}),
    "keyboard": ("input", "medium", {"text": ("str", 2000, False), "keys": ("str", 100, False)}),
}
_CTRL = re.compile(r"[\x00]")


def validate_args(kind: str, args: dict | None) -> dict:
    """Nettoie les arguments : types, bornes, aucun argument inconnu. Lève DeviceError sinon."""
    if kind not in KINDS:
        raise DeviceError(f"Action inconnue « {kind} ». Actions possibles : {', '.join(KINDS)}.")
    spec = KINDS[kind][2]
    args = dict(args or {})
    unknown = set(args) - set(spec)
    if unknown:
        raise DeviceError(f"Arguments non reconnus pour {kind} : {', '.join(sorted(unknown))}.")
    out: dict = {}
    for name, rule in spec.items():
        value = args.get(name)
        typ = rule[0]
        if typ == "str":
            _, maxlen, required = rule
            if value is None or value == "":
                if required:
                    raise DeviceError(f"« {name} » est obligatoire pour {kind}.")
                continue
            if not isinstance(value, str):
                raise DeviceError(f"« {name} » doit être du texte.")
            if len(value) > maxlen:
                raise DeviceError(f"« {name} » est trop long ({maxlen} caractères maximum).")
            if _CTRL.search(value):
                raise DeviceError(f"« {name} » contient un caractère interdit.")
            out[name] = value
        elif typ == "int":
            _, lo, hi, default = rule
            if value is None:
                if default is not None:
                    out[name] = default
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
                raise DeviceError(f"« {name} » doit être un nombre entier.")
            if not lo <= int(value) <= hi:
                raise DeviceError(f"« {name} » doit être entre {lo} et {hi}.")
            out[name] = int(value)
        elif typ == "bool":
            out[name] = bool(value) if value is not None else rule[1]
        elif typ == "enum":
            _, choices, default = rule
            value = default if value is None else value
            if value not in choices:
                raise DeviceError(f"« {name} » doit valoir : {', '.join(choices)}.")
            out[name] = value
    if kind == "mouse":
        if out["action"] in ("move", "click", "double_click", "right_click") and ("x" not in out or "y" not in out):
            raise DeviceError("La souris a besoin de x et y.")
        if out["action"] == "scroll" and not out.get("amount"):
            raise DeviceError("Le défilement a besoin d'un « amount » non nul.")
    if kind == "keyboard" and bool(out.get("text")) == bool(out.get("keys")):
        raise DeviceError("Le clavier prend soit « text » (taper), soit « keys » (touches, ex. ctrl+s), pas les deux.")
    return out


def describe(kind: str, args: dict) -> tuple[str, str]:
    """(résumé, détail) calculés par le serveur d'après les arguments réels : jamais un texte fourni par l'IA."""
    a = args
    if kind == "status":
        return "Voir l'état de la machine", ""
    if kind == "processes":
        return "Lister les programmes en cours", ""
    if kind == "list_dir":
        return f"Lister le dossier {a['path']}", ""
    if kind == "read_file":
        start = f" (à partir de l'octet {a['offset']})" if a.get("offset") else ""
        return f"Lire le fichier {a['path']}{start}", ""
    if kind == "search_files":
        where = "dans le contenu des fichiers de" if a.get("contents") else "dans les noms de fichiers de"
        return f"Chercher « {a['query']} » {where} {a['path']}", ""
    if kind == "write_file":
        verb = {"create": "Créer", "overwrite": "Remplacer", "append": "Ajouter à"}[a["mode"]]
        n = len(a["content"])
        return f"{verb} {a['path']} ({n} caractères)", a["content"]
    if kind == "make_dir":
        return f"Créer le dossier {a['path']}", ""
    if kind == "move":
        return f"Déplacer {a['src']} vers {a['dst']}", ""
    if kind == "delete":
        return f"Supprimer {a['path']} (mis à la corbeille de KIRA, récupérable)", ""
    if kind == "run_command":
        where = f" dans {a['cwd']}" if a.get("cwd") else ""
        return f"Lancer une commande{where}", a["command"]
    if kind == "run_python":
        return "Lancer un script Python", a["code"]
    if kind == "open":
        return f"Ouvrir {a['target']}", ""
    if kind == "screenshot":
        return "Prendre une capture de l'écran", ""
    if kind == "mouse":
        act = {"move": "Déplacer la souris", "click": "Cliquer", "double_click": "Double-cliquer",
               "right_click": "Clic droit", "scroll": "Faire défiler"}[a["action"]]
        pos = f" en ({a['x']}, {a['y']})" if "x" in a else ""
        amount = f" de {a['amount']}" if a["action"] == "scroll" else ""
        return f"{act}{pos}{amount}", ""
    if kind == "keyboard":
        if a.get("text"):
            return "Taper du texte au clavier", a["text"]
        return f"Appuyer sur {a['keys']}", ""
    return kind, ""


# ---------------------------------------------------------------- appareils
def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _utc(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        d = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _ago(seconds: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat(timespec="seconds")


def global_paused() -> bool:
    return get_db().kv_get("devices_paused") == "1"


def set_global_pause(paused: bool) -> None:
    get_db().kv_set("devices_paused", "1" if paused else "0")
    if paused:
        get_db().run(
            "UPDATE device_actions SET status = 'cancelled', finished_at = ?, result = 'Mise en pause générale.' "
            "WHERE status IN ('pending', 'queued')",
            [now_iso()],
        )
    audit.log("devices_pause_all" if paused else "devices_resume_all", {}, actor="owner")


def _policy(row: dict) -> dict:
    stored = jload(row.get("policy"), {})
    merged = dict(DEFAULT_POLICY)
    for cat, mode in (stored or {}).items():
        if cat in CATEGORIES and mode in MODES:
            merged[cat] = mode
    for cat in NEVER_AUTO:
        if merged[cat] == "auto":
            merged[cat] = "ask"
    return merged


def _public(row: dict) -> dict:
    caps = jload(row.get("caps"), {})
    seen = _utc(row.get("last_seen"))
    online = bool(seen and (datetime.now(timezone.utc) - seen).total_seconds() <= ONLINE_WINDOW_S)
    return {
        "id": row["id"],
        "name": row["name"],
        "platform": row["platform"],
        "online": online and not row["revoked"],
        "last_seen": row.get("last_seen"),
        "paused": bool(row["paused"]),
        "revoked": bool(row["revoked"]),
        "policy": _policy(row),
        "enabled": caps.get("enabled", []),
        "available": caps.get("available", {}),
        "roots": caps.get("roots", []),
        "local_pause": bool(caps.get("local_pause")),
        "info": jload(row.get("info"), {}),
        "agent_version": row.get("agent_version") or "",
        "created_at": row["created_at"],
    }


def get_device(device_id: int, include_revoked: bool = False) -> dict | None:
    row = get_db().q1("SELECT * FROM devices WHERE id = ?", [device_id])
    if not row or (row["revoked"] and not include_revoked):
        return None
    return row


def list_devices() -> list[dict]:
    rows = get_db().q("SELECT * FROM devices WHERE revoked = 0 ORDER BY id")
    return [_public(r) for r in rows]


def find_device(ref) -> dict | None:
    """Trouve un appareil par numéro ou par nom (sans tenir compte de la casse). None s'il y en a 0 ou plusieurs."""
    rows = get_db().q("SELECT * FROM devices WHERE revoked = 0 ORDER BY id")
    if ref in (None, ""):
        return rows[0] if len(rows) == 1 else None
    text = str(ref).strip().lower()
    exact = [r for r in rows if str(r["id"]) == text or r["name"].lower() == text]
    if len(exact) == 1:
        return exact[0]
    partial = [r for r in rows if text in r["name"].lower()]
    return partial[0] if len(partial) == 1 else None


def set_policy(device_id: int, policy: dict) -> dict:
    row = get_device(device_id)
    if not row:
        raise DeviceError("Appareil introuvable.")
    current = _policy(row)
    for cat, mode in (policy or {}).items():
        if cat not in CATEGORIES:
            raise DeviceError(f"Catégorie inconnue « {cat} ».")
        if mode not in MODES:
            raise DeviceError("Le mode doit être auto, ask ou deny.")
        if mode == "auto" and cat in NEVER_AUTO:
            raise DeviceError("Supprimer demande toujours ton accord : ce réglage ne peut pas être « libre ».")
        current[cat] = mode
    get_db().update("devices", device_id, {"policy": jdump(current)})
    audit.log("device_policy_changed", {"device": device_id, "policy": current}, actor="owner")
    return _public(get_device(device_id))


def rename(device_id: int, name: str) -> dict:
    name = " ".join((name or "").split())[:60]
    if not name or not get_device(device_id):
        raise DeviceError("Nom vide ou appareil introuvable.")
    get_db().update("devices", device_id, {"name": name})
    return _public(get_device(device_id))


def set_paused(device_id: int, paused: bool) -> dict:
    if not get_device(device_id):
        raise DeviceError("Appareil introuvable.")
    get_db().update("devices", device_id, {"paused": 1 if paused else 0})
    if paused:
        get_db().run(
            "UPDATE device_actions SET status = 'cancelled', finished_at = ?, result = 'Appareil mis en pause.' "
            "WHERE device_id = ? AND status IN ('pending', 'queued')",
            [now_iso(), device_id],
        )
    audit.log("device_paused" if paused else "device_resumed", {"device": device_id}, actor="owner")
    return _public(get_device(device_id))


def revoke(device_id: int) -> None:
    """Coupe l'appareil tout de suite : son jeton ne vaut plus rien et ses actions en cours sont annulées."""
    row = get_device(device_id)
    if not row:
        raise DeviceError("Appareil introuvable.")
    db = get_db()
    db.update("devices", device_id, {"revoked": 1})
    db.run(
        "UPDATE device_actions SET status = 'cancelled', finished_at = ?, result = 'Appareil révoqué.' "
        "WHERE device_id = ? AND status IN ('pending', 'queued', 'running')",
        [now_iso(), device_id],
    )
    audit.log("device_revoked", {"device": device_id, "name": row["name"]}, actor="owner")


# ---------------------------------------------------------------- appairage
_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # sans I, L, O, 0, 1 : pas de confusion à la saisie


def _code_hash(code: str) -> str:
    return hashlib.sha256(("kira-pair|" + re.sub(r"[^A-Z0-9]", "", (code or "").upper())).encode("utf-8")).hexdigest()


def create_pair_code() -> dict:
    db = get_db()
    db.run("DELETE FROM pair_codes WHERE used = 1 OR expires_at < ?", [now_iso()])
    live = int((db.q1("SELECT COUNT(*) AS n FROM pair_codes") or {"n": 0})["n"])
    if live >= 3:
        db.run("DELETE FROM pair_codes")  # on repart d'un seul code valable
    code = "".join(secrets.choice(_ALPHABET) for _ in range(8))
    expires = (datetime.now(timezone.utc) + timedelta(seconds=PAIR_TTL_S)).isoformat(timespec="seconds")
    db.insert("pair_codes", {"code_hash": _code_hash(code), "expires_at": expires, "used": 0, "created_at": now_iso()})
    audit.log("device_pair_code", {}, actor="owner")
    return {"code": f"{code[:4]}-{code[4:]}", "expires_in": PAIR_TTL_S}


def _caps_blob(info: dict) -> str:
    enabled = [c for c in (info.get("enabled") or []) if c in CATEGORIES]
    available = info.get("available") if isinstance(info.get("available"), dict) else {}
    roots = [str(r)[:300] for r in (info.get("roots") or [])][:30]
    return jdump({
        "enabled": enabled,
        "available": {str(k)[:40]: bool(v) if isinstance(v, bool) else str(v)[:80] for k, v in list(available.items())[:20]},
        "roots": roots,
        "local_pause": bool(info.get("local_pause")),
    })


def _info_blob(info: dict) -> str:
    raw = info.get("system") if isinstance(info.get("system"), dict) else {}
    return jdump({str(k)[:40]: (v if isinstance(v, (int, float, bool)) else str(v)[:200]) for k, v in list(raw.items())[:25]})


def pair(code: str, name: str, platform: str, info: dict, agent_version: str = "") -> dict:
    db = get_db()
    row = db.q1("SELECT * FROM pair_codes WHERE code_hash = ? AND used = 0 AND expires_at > ?", [_code_hash(code), now_iso()])
    if not row or db.run("UPDATE pair_codes SET used = 1 WHERE id = ? AND used = 0", [row["id"]]) != 1:
        raise DeviceError("Code invalide ou expiré. Génère-en un nouveau dans l'appli.")
    token = "kdev_" + secrets.token_urlsafe(32)
    clean_name = " ".join((name or "").split())[:60] or (platform or "Appareil")
    platform = (platform or "")[:20]
    device_id = db.insert("devices", {
        "name": clean_name, "platform": platform, "token_hash": _hash(token), "caps": _caps_blob(info or {}),
        "policy": "{}", "paused": 0, "revoked": 0, "info": _info_blob(info or {}), "agent_version": (agent_version or "")[:20],
        "created_at": now_iso(), "last_seen": now_iso(),
    })
    audit.log("device_paired", {"device": device_id, "name": clean_name, "platform": platform}, actor="owner")
    return {"device_id": device_id, "token": token, "name": clean_name}


def authenticate(header: str) -> dict | None:
    """``Authorization: Device <jeton>``. Un jeton de propriétaire (Bearer) n'ouvre jamais cette porte, et inversement."""
    parts = (header or "").split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "device" or not parts[1].startswith("kdev_"):
        return None
    return get_db().q1("SELECT * FROM devices WHERE token_hash = ? AND revoked = 0", [_hash(parts[1].strip())])


# ---------------------------------------------------------------- décision de politique
def decide(device: dict, category: str, tainted: bool = False) -> tuple[str, str]:
    """(mode, raison). ``tainted`` : le tour en cours a lu du contenu venu d'Internet, donc potentiellement piégé."""
    if device["revoked"]:
        return "deny", "Cet appareil a été révoqué."
    if global_paused():
        return "deny", "Tous les appareils sont en pause."
    if device["paused"]:
        return "deny", "Cet appareil est en pause."
    caps = jload(device.get("caps"), {})
    if caps.get("local_pause"):
        return "deny", "L'appareil est en pause (réglage local sur la machine)."
    if category not in (caps.get("enabled") or []):
        return "deny", f"« {CATEGORIES[category]} » est désactivé sur la machine elle-même."
    mode = _policy(device)[category]
    if mode == "auto" and tainted and category != "monitor":
        return "ask", "Contenu venu d'Internet lu pendant ce tour : accord demandé par précaution."
    return mode, ""


def _view_action(row: dict, full: bool = True) -> dict:
    args = jload(row.get("args"), {})
    try:
        summary, detail = describe(row["kind"], args)
    except Exception:  # noqa: BLE001
        summary, detail = row.get("summary") or row["kind"], ""
    result = row.get("result") or ""
    return {
        "id": row["id"],
        "device_id": row["device_id"],
        "device_name": row.get("device_name") or "",
        "kind": row["kind"],
        "category": row["category"],
        "category_label": CATEGORIES.get(row["category"], row["category"]),
        "risk": row["risk"],
        "summary": summary,
        "detail": detail,
        "status": row["status"],
        "requested_by": row["requested_by"],
        "decided_by": row.get("decided_by"),
        "reason": row.get("reason") or "",
        "result": result if full else result[:300],
        "files": [files.url(f) for f in jload(row.get("result_files"), [])],
        "created_at": row["created_at"],
        "finished_at": row.get("finished_at"),
    }


def get_action(action_id: int) -> dict | None:
    row = get_db().q1(
        "SELECT a.*, d.name AS device_name FROM device_actions a LEFT JOIN devices d ON d.id = a.device_id WHERE a.id = ?",
        [action_id],
    )
    return _view_action(row) if row else None


def request_action(device_id: int, kind: str, args: dict | None, requested_by: str = "kira",
                   conversation_id: str = "", tainted: bool = False) -> dict:
    """Seul chemin pour demander une action à une machine. Applique la politique et journalise."""
    device = get_device(device_id)
    if not device:
        raise DeviceError("Appareil introuvable ou révoqué.")
    clean = validate_args(kind, args)
    category, risk, _ = KINDS[kind]
    summary, _detail = describe(kind, clean)
    mode, reason = decide(device, category, tainted)
    status = {"auto": "queued", "ask": "pending", "deny": "denied"}[mode]
    now = now_iso()
    action_id = get_db().insert("device_actions", {
        "device_id": device_id, "kind": kind, "args": jdump(clean), "category": category, "risk": risk, "summary": summary[:500],
        "status": status, "requested_by": requested_by, "conversation_id": conversation_id or "",
        "decided_by": "policy" if status != "pending" else None, "reason": reason,
        "result": reason if status == "denied" else None, "result_files": "[]",
        "created_at": now, "decided_at": now if status != "pending" else None,
        "finished_at": now if status == "denied" else None,
    })
    audit.log("device_action_requested", {"action": action_id, "device": device_id, "kind": kind, "summary": summary[:200],
                                          "mode": mode, "tainted": tainted}, actor=requested_by)
    return get_action(action_id)


def wait_for(action_id: int, timeout: float, step: float = 0.4) -> dict:
    """Attend (au plus ``timeout`` secondes) que l'action soit terminée ; renvoie son état courant."""
    deadline = time.time() + max(0.0, timeout)
    action = get_action(action_id)
    while action and action["status"] not in TERMINAL and time.time() < deadline:
        time.sleep(step)
        expire_stale()
        action = get_action(action_id)
    return action


def pending_actions() -> list[dict]:
    expire_stale()
    rows = get_db().q(
        "SELECT a.*, d.name AS device_name FROM device_actions a LEFT JOIN devices d ON d.id = a.device_id "
        "WHERE a.status = 'pending' ORDER BY a.id"
    )
    return [_view_action(r, full=False) for r in rows]


def list_actions(device_id: int | None = None, limit: int = 30) -> list[dict]:
    if device_id is None:
        rows = get_db().q(
            "SELECT a.*, d.name AS device_name FROM device_actions a LEFT JOIN devices d ON d.id = a.device_id "
            "ORDER BY a.id DESC LIMIT ?", [limit])
    else:
        rows = get_db().q(
            "SELECT a.*, d.name AS device_name FROM device_actions a LEFT JOIN devices d ON d.id = a.device_id "
            "WHERE a.device_id = ? ORDER BY a.id DESC LIMIT ?", [device_id, limit])
    return [_view_action(r, full=False) for r in rows]


def approve(action_id: int) -> dict:
    expire_stale()
    row = get_db().q1("SELECT * FROM device_actions WHERE id = ?", [action_id])
    if not row:
        raise DeviceError("Demande introuvable.")
    if row["status"] != "pending":
        raise DeviceError(f"Cette demande est déjà « {row['status']} ».")
    device = get_device(row["device_id"])
    if not device or device["paused"] or global_paused():
        get_db().update("device_actions", action_id, {"status": "cancelled", "finished_at": now_iso(),
                                                      "result": "Appareil indisponible ou en pause."})
        raise DeviceError("L'appareil est indisponible ou en pause : demande annulée.")
    if get_db().run("UPDATE device_actions SET status = 'queued', decided_by = 'owner', decided_at = ? "
                    "WHERE id = ? AND status = 'pending'", [now_iso(), action_id]) != 1:
        raise DeviceError("Cette demande n'est plus en attente.")
    audit.log("device_action_approved", {"action": action_id, "kind": row["kind"]}, actor="owner")
    return get_action(action_id)


def deny(action_id: int) -> dict:
    expire_stale()
    row = get_db().q1("SELECT * FROM device_actions WHERE id = ?", [action_id])
    if not row:
        raise DeviceError("Demande introuvable.")
    if get_db().run("UPDATE device_actions SET status = 'denied', decided_by = 'owner', decided_at = ?, finished_at = ?, "
                    "result = 'Refusé par Brice.' WHERE id = ? AND status = 'pending'", [now_iso(), now_iso(), action_id]) != 1:
        raise DeviceError(f"Cette demande est déjà « {row['status']} ».")
    audit.log("device_action_denied", {"action": action_id, "kind": row["kind"]}, actor="owner")
    return get_action(action_id)


def expire_stale() -> None:
    db = get_db()
    now = now_iso()
    db.run("UPDATE device_actions SET status = 'expired', finished_at = ?, result = 'Sans réponse à temps : demande expirée.' "
           "WHERE status = 'pending' AND created_at < ?", [now, _ago(PENDING_TTL_S)])
    db.run("UPDATE device_actions SET status = 'expired', finished_at = ?, result = 'La machine n''a pas répondu à temps.' "
           "WHERE status = 'queued' AND decided_at < ?", [now, _ago(QUEUED_TTL_S)])
    db.run("UPDATE device_actions SET status = 'failed', finished_at = ?, result = 'Délai dépassé côté machine.' "
           "WHERE status = 'running' AND started_at < ?", [now, _ago(RUNNING_TTL_S)])


# ---------------------------------------------------------------- côté machine
def heartbeat(device_id: int, info: dict) -> None:
    get_db().update("devices", device_id, {
        "last_seen": now_iso(), "caps": _caps_blob(info or {}), "info": _info_blob(info or {}),
        "agent_version": str((info or {}).get("version") or "")[:20],
    })


def touch(device: dict) -> dict:
    """Signe de vie léger pendant qu'une action dure : met à jour « vu à » et dit si Brice a mis la machine en pause."""
    get_db().update("devices", device["id"], {"last_seen": now_iso()})
    fresh = get_device(device["id"])
    return {"paused": bool(global_paused() or not fresh or fresh["paused"])}


def _claim_next(device_id: int) -> dict | None:
    db = get_db()
    row = db.q1("SELECT id FROM device_actions WHERE device_id = ? AND status = 'queued' ORDER BY id LIMIT 1", [device_id])
    if not row:
        return None
    if db.run("UPDATE device_actions SET status = 'running', started_at = ? WHERE id = ? AND status = 'queued'",
              [now_iso(), row["id"]]) != 1:
        return None
    act = db.q1("SELECT * FROM device_actions WHERE id = ?", [row["id"]])
    return {"id": act["id"], "kind": act["kind"], "args": jload(act["args"], {})}


def poll(device: dict, info: dict, wait: float = 20.0) -> dict:
    """Appel de la machine (scrutation sortante). Renvoie la prochaine action autorisée, ou rien après ``wait`` secondes."""
    heartbeat(device["id"], info)
    expire_stale()
    deadline = time.time() + max(0.0, wait)
    while True:
        fresh = get_device(device["id"])
        if not fresh:
            return {"action": None, "revoked": True}
        if global_paused() or fresh["paused"] or (info or {}).get("local_pause"):
            return {"action": None, "paused": True}
        action = _claim_next(device["id"])
        if action:
            return {"action": action}
        if time.time() >= deadline:
            return {"action": None}
        time.sleep(0.5)


_PNG = b"\x89PNG\r\n\x1a\n"
_JPEG = b"\xff\xd8\xff"


def submit_result(device: dict, action_id: int, ok: bool, output: str, images: list | None = None) -> dict:
    db = get_db()
    row = db.q1("SELECT * FROM device_actions WHERE id = ? AND device_id = ?", [action_id, device["id"]])
    if not row or row["status"] != "running":
        raise DeviceError("Action inconnue ou déjà terminée.")
    saved: list[str] = []
    import base64

    for i, img in enumerate((images or [])[:3]):
        try:
            raw = base64.b64decode(str((img or {}).get("b64", "")), validate=True)
        except (ValueError, TypeError):
            continue
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            continue
        mime = "image/png" if raw.startswith(_PNG) else "image/jpeg" if raw.startswith(_JPEG) else ""
        if not mime:
            continue
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        saved.append(files.save(f"capture-{device['id']}-{stamp}-{i + 1}.{'png' if mime.endswith('png') else 'jpg'}", mime, raw))
    text = str(output or "")[:MAX_RESULT_CHARS]
    db.update("device_actions", action_id, {
        "status": "done" if ok else "failed", "result": text, "result_files": jdump(saved), "finished_at": now_iso(),
    })
    audit.log("device_action_done", {"action": action_id, "ok": bool(ok), "kind": row["kind"], "device": device["id"]},
              actor=f"device:{device['id']}")
    return get_action(action_id)


def result_images(action: dict) -> list[dict]:
    """Images d'une action terminée, prêtes à être montrées à un modèle qui voit les images."""
    import base64

    out = []
    for url in action.get("files", []):
        row = files.get(url.rsplit("/", 1)[-1])
        if row:
            out.append({"mime": row["mime"], "b64": base64.b64encode(row["data"]).decode("ascii")})
    return out


def cleanup(action_days: int = 90, capture_days: int = 7, result_days: int = 7) -> dict:
    """Ce que les machines ont renvoyé (contenu de fichiers, sorties, captures) ne reste pas : 7 jours. Le journal garde 90 jours."""
    db = get_db()
    now = datetime.now(timezone.utc)
    cutoff_a = (now - timedelta(days=action_days)).isoformat(timespec="seconds")
    cutoff_c = (now - timedelta(days=capture_days)).isoformat(timespec="seconds")
    cutoff_r = (now - timedelta(days=result_days)).isoformat(timespec="seconds")
    a = db.run("DELETE FROM device_actions WHERE finished_at IS NOT NULL AND finished_at < ?", [cutoff_a])
    r = db.run("UPDATE device_actions SET result = '(effacé)', args = '{}', result_files = '[]' "
               "WHERE finished_at IS NOT NULL AND finished_at < ? AND status IN ('done', 'failed') AND result != '(effacé)'", [cutoff_r])
    c = db.run("DELETE FROM files WHERE name LIKE 'capture-%' AND created_at < ?", [cutoff_c])
    return {"actions_deleted": a, "results_erased": r, "captures_deleted": c}


def prompt_summary() -> str:
    """Quelques lignes pour le contexte de KIRA : quels appareils existent et ce qu'il peut y faire sans demander."""
    devices = list_devices()
    if not devices:
        return ""
    lines = []
    for d in devices:
        free = [CATEGORIES[c].lower() for c in CATEGORIES if d["policy"].get(c) == "auto" and c in d["enabled"]]
        ask = [CATEGORIES[c].lower() for c in CATEGORIES if d["policy"].get(c) == "ask" and c in d["enabled"]]
        state = "en ligne" if d["online"] else "hors ligne"
        if d["paused"] or d["local_pause"]:
            state += ", en pause"
        lines.append(f"- {d['name']} (n°{d['id']}, {d['platform'] or '?'}) : {state}. Libre : {', '.join(free) or 'rien'}. "
                     f"Sur accord de Brice : {', '.join(ask) or 'rien'}.")
    return "\n".join(lines)
