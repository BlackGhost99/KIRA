#!/usr/bin/env python3
"""KIRA Core : le « cœur » de KIRA sur une machine (Windows, Linux, macOS, Android/Termux).

Un seul fichier, bibliothèque standard uniquement (Python 3.8 ou plus). Il appelle le serveur KIRA (jamais l'inverse :
aucun port ouvert sur ta machine), reçoit les actions que TU as autorisées dans l'application, les exécute ici, puis
renvoie le résultat.

Ce que la machine décide elle-même, et que le serveur ne peut pas élargir :
  * quelles catégories de pouvoirs sont actives (``allow`` / ``deny``) ;
  * dans quels dossiers KIRA peut lire et écrire (``roots``), avec une liste de fichiers sensibles toujours interdits ;
  * la pause locale (``pause`` / ``resume``), ou le coin de l'écran si la souris est pilotée.

Commandes :
  pair --server URL --code CODE [--name NOM] [--full] [--root DOSSIER]   appairer cette machine
  run                         tourner en continu (démarrage manuel)
  install / uninstall         démarrage automatique à l'ouverture de session
  status                      état, pouvoirs, dossiers
  allow CAT / deny CAT        activer / couper un pouvoir sur CETTE machine
  roots list | add D | remove D
  pause / resume              pause locale immédiate
  trash list | restore ID     corbeille de KIRA (rien n'est supprimé pour de bon)
  unpair                      oublier le serveur (pense aussi à « Retirer » dans l'application)
"""
from __future__ import annotations

import argparse
import base64
import collections
import fnmatch
import importlib.util
import io
import json
import locale
import logging
import logging.handlers
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

VERSION = "1.0.0"

CATEGORIES = {
    "monitor": "Surveiller l'état de la machine",
    "fs_read": "Lire des fichiers",
    "fs_write": "Écrire des fichiers",
    "fs_delete": "Supprimer des fichiers",
    "exec": "Lancer des commandes",
    "screen": "Voir l'écran",
    "input": "Piloter souris et clavier",
}
DEFAULT_ENABLED = ["monitor", "fs_read", "fs_write", "fs_delete"]  # confinés aux dossiers autorisés
SENSITIVE_CATEGORIES = ("exec", "screen", "input")  # désactivés tant que tu ne les actives pas toi-même ici

# Catalogue des actions : même forme que le serveur (tests/test_core_agent.py vérifie qu'ils restent identiques).
_PATH = ("str", 2000, True)
# Brice doit voir EN ENTIER ce qu'il approuve : mêmes plafonds que le serveur (voir app/devices.py).
MAX_SHOWN_WRITE = 20_000
MAX_SHOWN_CODE = 6_000
KINDS = {
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

# Jamais accessibles, même dans un dossier autorisé.
DENY_DIRS = {".ssh", ".gnupg", ".aws", ".kube", ".azure", ".docker", ".kira-core", ".mozilla", ".password-store"}
DENY_FILES = [
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.kdbx", "*.keystore", "*.jks", "id_rsa*", "id_ed25519*",
    "id_ecdsa*", "id_dsa*", ".netrc", "_netrc", ".npmrc", ".pypirc", ".git-credentials", "credentials", "credentials.json",
    "secrets.*", "login data", "cookies", "cookies.sqlite", "key4.db", "logins.json",
]
EXEC_EXTENSIONS = {".exe", ".bat", ".cmd", ".com", ".scr", ".msi", ".ps1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".jar",
                   ".sh", ".lnk", ".reg", ".hta", ".dll", ".app", ".command", ".pif", ".cpl", ".appimage", ".apk"}
# Écrire un programme ou un fichier de démarrage revient à pouvoir lancer des commandes : refusé tant que « exec » n'est pas
# activé sur CETTE machine (sinon écrire un script contournerait le plafond local).
PROGRAM_EXTENSIONS = EXEC_EXTENSIONS | {".desktop", ".service", ".timer", ".plist", ".pth", ".psm1", ".psd1"}
AUTORUN_NAMES = {".bashrc", ".bash_profile", ".bash_login", ".bash_logout", ".profile", ".zshrc", ".zprofile", ".zshenv",
                 ".zlogin", ".xinitrc", ".xprofile", "autorun.inf", "sitecustomize.py", "usercustomize.py", "profile.ps1",
                 "crontab"}
AUTORUN_DIRS = {".husky", "launchagents", "launchdaemons", "autostart", "startup", "init.d", "cron.d", "cron.daily",
                "cron.hourly", "systemd"}
MAX_OUTPUT = 20_000
SHOT_MAX_WIDTH = 1280
SHOT_MAX_BYTES = 3_500_000
TRASH_KEEP_DAYS = 30


class Refused(Exception):
    """Refus décidé par la machine (message affiché tel quel à Brice)."""


class ApiFail(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ dossiers et configuration
def home_dir():
    return os.environ.get("KIRA_CORE_HOME") or os.path.join(os.path.expanduser("~"), ".kira-core")


def default_root():
    return os.path.join(os.path.expanduser("~"), "KIRA")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def platform_name():
    if "com.termux" in os.environ.get("PREFIX", "") or hasattr(sys, "getandroidapilevel"):
        return "android"
    return {"win32": "windows", "darwin": "mac"}.get(sys.platform, "linux")


def pause_file():
    return os.path.join(home_dir(), "PAUSE")


def is_paused():
    return os.path.exists(pause_file())


def set_paused(paused):
    os.makedirs(home_dir(), exist_ok=True)
    if paused:
        with open(pause_file(), "w", encoding="utf-8") as f:
            f.write(now_iso())
    elif os.path.exists(pause_file()):
        os.remove(pause_file())


class Config:
    """~/.kira-core/config.json (droits 0600 : il contient le jeton de cette machine)."""

    def __init__(self, path=None):
        self.path = path or os.path.join(home_dir(), "config.json")
        self.data = {"server": "", "token": "", "name": "", "device_id": 0, "enabled": list(DEFAULT_ENABLED), "roots": []}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.data.update(loaded)

    @property
    def paired(self):
        return bool(self.data.get("server") and self.data.get("token"))

    @property
    def enabled(self):
        return [c for c in self.data.get("enabled", []) if c in CATEGORIES]

    @property
    def roots(self):
        return list(self.data.get("roots", []))

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass


# ------------------------------------------------------------------ garde-fou des chemins
def _real(path):
    return os.path.realpath(os.path.expanduser(path))


def _norm(path):
    return os.path.normcase(path)


def _within(root, path):
    try:
        return os.path.commonpath([_norm(root), _norm(path)]) == _norm(root)
    except ValueError:  # lecteurs différents (Windows)
        return False


class Guard:
    """Décide si un chemin est permis : dans un dossier autorisé, sans lien sortant, hors liste sensible."""

    def __init__(self, roots):
        self.roots = [_real(r) for r in roots]
        self.home = _real(home_dir())

    def _denied(self, real):
        base = os.path.basename(real).lower()
        parts = {p.lower() for p in re.split(r"[\\/]+", real) if p}
        if parts & DENY_DIRS:
            return True
        return any(fnmatch.fnmatch(base, pat) for pat in DENY_FILES)

    def resolve(self, path, is_root_ok=True):
        if not self.roots:
            raise Refused("Aucun dossier n'est autorisé sur cette machine (python kira_core.py roots add DOSSIER).")
        raw = os.path.expanduser(path)
        if os.name == "nt" and ":" in raw[2:]:
            raise Refused("Chemin refusé (flux de données alternatif).")
        if not os.path.isabs(raw):
            raw = os.path.join(self.roots[0], raw)
        real = _real(raw)
        if _within(self.home, real):
            raise Refused("Le dossier de KIRA Core lui-même est interdit (jeton et réglages de cette machine).")
        if not any(_within(r, real) for r in self.roots):
            raise Refused(f"Hors des dossiers autorisés : {real}. Dossiers permis : {', '.join(self.roots)}.")
        if self._denied(real):
            raise Refused("Fichier ou dossier sensible (clés, mots de passe, jetons) : toujours interdit.")
        if not is_root_ok and any(_norm(real) == _norm(r) for r in self.roots):
            raise Refused("Ce dossier est une racine autorisée : on ne la touche pas.")
        return real

    def skip_in_walk(self, real):
        return self._denied(real) or _within(self.home, real)


# ------------------------------------------------------------------ validation des arguments (copie du serveur)
def validate_args(kind, args):
    if kind not in KINDS:
        raise Refused(f"Action inconnue : {kind}.")
    spec = KINDS[kind][2]
    args = dict(args or {})
    unknown = set(args) - set(spec)
    if unknown:
        raise Refused(f"Arguments non reconnus : {', '.join(sorted(unknown))}.")
    out = {}
    for name, rule in spec.items():
        value = args.get(name)
        typ = rule[0]
        if typ == "str":
            _, maxlen, required = rule
            if value is None or value == "":
                if required:
                    raise Refused(f"« {name} » est obligatoire.")
                continue
            if not isinstance(value, str) or len(value) > maxlen or "\x00" in value:
                raise Refused(f"« {name} » invalide.")
            out[name] = value
        elif typ == "int":
            _, lo, hi, default = rule
            if value is None:
                if default is not None:
                    out[name] = default
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or not lo <= int(value) <= hi:
                raise Refused(f"« {name} » doit être un entier entre {lo} et {hi}.")
            out[name] = int(value)
        elif typ == "bool":
            out[name] = bool(value) if value is not None else rule[1]
        elif typ == "enum":
            _, choices, default = rule
            value = default if value is None else value
            if value not in choices:
                raise Refused(f"« {name} » doit valoir : {', '.join(choices)}.")
            out[name] = value
    if kind == "mouse":
        if out["action"] != "scroll" and ("x" not in out or "y" not in out):
            raise Refused("La souris a besoin de x et y.")
        if out["action"] == "scroll" and not out.get("amount"):
            raise Refused("Le défilement a besoin d'un « amount » non nul.")
    if kind == "keyboard" and bool(out.get("text")) == bool(out.get("keys")):
        raise Refused("Le clavier prend soit « text », soit « keys », pas les deux.")
    return out


class Result:
    def __init__(self, ok, output, images=None):
        self.ok = ok
        self.output = output
        self.images = images or []


def _clip(text, limit=MAX_OUTPUT):
    if len(text) <= limit:
        return text
    head, tail = int(limit * 0.6), int(limit * 0.3)
    return f"{text[:head]}\n…[{len(text) - head - tail} caractères omis]…\n{text[-tail:]}"


def _has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _fmt_bytes(n):
    n = float(n)
    for unit in ("o", "Ko", "Mo", "Go", "To"):
        if n < 1024 or unit == "To":
            return f"{n:.0f} {unit}" if unit == "o" else f"{n:.1f} {unit}"
        n /= 1024


# ------------------------------------------------------------------ corbeille
class Trash:
    def __init__(self):
        self.base = os.path.join(home_dir(), "trash")

    def put(self, real, reason, copy=False):
        os.makedirs(self.base, exist_ok=True)
        entry = os.path.join(self.base, datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(2).hex())
        os.makedirs(entry)
        data = os.path.join(entry, "data")
        if copy:
            if os.path.isdir(real):
                shutil.copytree(real, data, symlinks=True)
            else:
                shutil.copy2(real, data)
        else:
            shutil.move(real, data)
        with open(os.path.join(entry, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"original": real, "when": now_iso(), "reason": reason}, f, ensure_ascii=False)
        return os.path.basename(entry)

    def entries(self):
        out = []
        if os.path.isdir(self.base):
            for name in sorted(os.listdir(self.base)):
                try:
                    with open(os.path.join(self.base, name, "meta.json"), encoding="utf-8") as f:
                        meta = json.load(f)
                    out.append({"id": name, **meta})
                except (OSError, ValueError):
                    continue
        return out

    def restore(self, entry_id):
        if not re.match(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$", entry_id or ""):
            raise Refused("Identifiant de corbeille invalide.")
        entry = os.path.join(self.base, entry_id)
        try:
            with open(os.path.join(entry, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            raise Refused("Entrée introuvable dans la corbeille.")
        target = meta["original"]
        if os.path.lexists(target):
            raise Refused(f"{target} existe déjà : déplace-le d'abord.")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.move(os.path.join(entry, "data"), target)
        shutil.rmtree(entry, ignore_errors=True)
        return target

    def purge(self, days=TRASH_KEEP_DAYS):
        cutoff = time.time() - days * 86400
        for name in (os.listdir(self.base) if os.path.isdir(self.base) else []):
            path = os.path.join(self.base, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass


# ------------------------------------------------------------------ le cœur : exécution des actions
class Core:
    def __init__(self, cfg, log=None):
        self.cfg = cfg
        self.log = log or logging.getLogger("kira-core")
        self.guard = Guard(cfg.roots)
        self.trash = Trash()
        self.abort = threading.Event()   # levé quand Brice met la machine en pause pendant une action
        self.last_shot = None            # {"sent": (w, h)} : sert à convertir les coordonnées de la capture en écran
        self._input_times = collections.deque(maxlen=60)

    def reload(self):
        """Relit les pouvoirs et dossiers modifiés en local (allow/deny/roots) : effet immédiat, sans redémarrer.
        Le serveur et le jeton ne changent jamais ici."""
        try:
            fresh = Config(self.cfg.path)
        except (OSError, ValueError):
            return
        self.cfg.data["enabled"] = fresh.data.get("enabled", [])
        self.cfg.data["roots"] = fresh.data.get("roots", [])
        self.guard = Guard(self.cfg.roots)

    # -- état envoyé au serveur ------------------------------------------------
    def available(self):
        shot = self._screenshot_backend()
        inp = self._input_backend()
        return {
            "screenshot": shot or False,
            "input": inp or False,
            "psutil": _has_module("psutil"),
            "pillow": _has_module("PIL"),
        }

    def info(self):
        return {
            "version": VERSION,
            "enabled": self.cfg.enabled,
            "roots": self.cfg.roots,
            "local_pause": is_paused(),
            "available": self.available(),
            "system": {
                "hostname": socket.gethostname(), "os": f"{platform.system()} {platform.release()}",
                "machine": platform.machine(), "python": platform.python_version(),
            },
        }

    # -- point d'entrée --------------------------------------------------------
    def execute(self, kind, args):
        if kind not in KINDS:
            return Result(False, f"Action inconnue : {kind}.")
        category = KINDS[kind][0]
        if category not in self.cfg.enabled:
            return Result(False, f"Refusé par la machine : « {CATEGORIES[category]} » n'est pas activé ici "
                                 f"(commande locale : python kira_core.py allow {category}).")
        if is_paused():
            return Result(False, "Refusé par la machine : pause locale active (python kira_core.py resume).")
        try:
            clean = validate_args(kind, args)
            return getattr(self, "do_" + kind)(clean)
        except Refused as exc:
            return Result(False, str(exc))
        except subprocess.TimeoutExpired:
            return Result(False, "Délai dépassé.")
        except Exception as exc:  # noqa: BLE001
            self.log.exception("action %s", kind)
            return Result(False, f"Erreur : {type(exc).__name__}: {exc}")

    # -- surveiller ------------------------------------------------------------
    def do_status(self, a):
        lines = [f"Machine : {socket.gethostname()} ({platform.platform()})", f"Python : {platform.python_version()}"]
        up = self._uptime()
        if up is not None:
            lines.append(f"En marche depuis : {int(up // 86400)} j {int(up % 86400 // 3600)} h {int(up % 3600 // 60)} min")
        lines.append(f"Processeur : {os.cpu_count() or '?'} cœurs" + (
            " · charge " + " ".join(f"{x:.2f}" for x in os.getloadavg()) if hasattr(os, "getloadavg") else ""))
        mem = self._memory()
        if mem:
            lines.append(f"Mémoire : {_fmt_bytes(mem[1])} libres sur {_fmt_bytes(mem[0])}")
        seen = set()
        for where in [os.path.expanduser("~")] + self.guard.roots:
            try:
                du = shutil.disk_usage(where)
            except OSError:
                continue
            key = (du.total, du.free)
            if key not in seen:
                seen.add(key)
                lines.append(f"Disque ({where}) : {_fmt_bytes(du.free)} libres sur {_fmt_bytes(du.total)}")
        battery = self._battery()
        if battery:
            lines.append(f"Batterie : {battery}")
        lines.append(f"Pouvoirs actifs ici : {', '.join(CATEGORIES[c] for c in self.cfg.enabled) or 'aucun'}")
        lines.append(f"Dossiers autorisés : {', '.join(self.guard.roots) or 'aucun'}")
        return Result(True, "\n".join(lines))

    @staticmethod
    def _uptime():
        try:
            if _has_module("psutil"):
                import psutil
                return time.time() - psutil.boot_time()
            if sys.platform.startswith("linux"):
                with open("/proc/uptime") as f:
                    return float(f.read().split()[0])
            if os.name == "nt":
                import ctypes
                return ctypes.windll.kernel32.GetTickCount64() / 1000.0
            if sys.platform == "darwin":
                out = subprocess.run(["sysctl", "-n", "kern.boottime"], capture_output=True, text=True, timeout=5).stdout
                m = re.search(r"sec = (\d+)", out)
                return time.time() - int(m.group(1)) if m else None
        except Exception:  # noqa: BLE001
            return None
        return None

    @staticmethod
    def _memory():
        """(total, disponible) en octets, ou None."""
        try:
            if _has_module("psutil"):
                import psutil
                vm = psutil.virtual_memory()
                return vm.total, vm.available
            if sys.platform.startswith("linux"):
                info = {}
                with open("/proc/meminfo") as f:
                    for line in f:
                        key, _, rest = line.partition(":")
                        info[key] = int(rest.split()[0]) * 1024
                return info["MemTotal"], info.get("MemAvailable", info.get("MemFree", 0))
            if os.name == "nt":
                import ctypes

                class MS(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

                st = MS()
                st.dwLength = ctypes.sizeof(MS)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
                return st.ullTotalPhys, st.ullAvailPhys
            if sys.platform == "darwin":
                total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout)
                return total, 0
        except Exception:  # noqa: BLE001
            return None
        return None

    @staticmethod
    def _battery():
        try:
            if _has_module("psutil"):
                import psutil
                b = psutil.sensors_battery()
                if b:
                    return f"{b.percent:.0f} %" + (" (sur secteur)" if b.power_plugged else " (sur batterie)")
            for cap in sorted(__import__("glob").glob("/sys/class/power_supply/BAT*/capacity")):
                status = open(os.path.join(os.path.dirname(cap), "status")).read().strip()
                return f"{open(cap).read().strip()} % ({status})"
            if shutil.which("termux-battery-status"):
                out = json.loads(subprocess.run(["termux-battery-status"], capture_output=True, text=True, timeout=10).stdout)
                return f"{out.get('percentage')} % ({out.get('status')})"
        except Exception:  # noqa: BLE001
            return None
        return None

    def do_processes(self, a):
        rows = []
        if _has_module("psutil"):
            import psutil
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                i = p.info
                rows.append((i["cpu_percent"] or 0.0, i["memory_percent"] or 0.0, i["pid"], i["name"] or "?"))
        elif os.name == "nt":
            out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=20,
                                 errors="replace").stdout
            for line in out.splitlines():
                cells = [c.strip('"') for c in line.split('","')]
                if len(cells) >= 5:
                    mem = re.sub(r"[^\d]", "", cells[4])
                    rows.append((0.0, float(mem or 0) / 1024.0, cells[1], cells[0]))  # Mo de mémoire dans la colonne %
        else:
            out = subprocess.run(["ps", "-A", "-o", "pid=,pcpu=,pmem=,comm="], capture_output=True, text=True, timeout=20,
                                 errors="replace").stdout
            for line in out.splitlines():
                parts = line.split(None, 3)
                if len(parts) == 4:
                    try:
                        rows.append((float(parts[1]), float(parts[2]), parts[0], parts[3]))
                    except ValueError:
                        pass
        rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
        unit = "Mo" if os.name == "nt" and not _has_module("psutil") else "% mém"
        lines = [f"{'PID':>7}  {'%CPU':>5}  {unit:>6}  Programme"]
        for cpu, mem, pid, name in rows[: a["limit"]]:
            lines.append(f"{pid!s:>7}  {cpu:5.1f}  {mem:6.1f}  {name}")
        return Result(True, "\n".join(lines))

    # -- fichiers : lecture ------------------------------------------------------
    def do_list_dir(self, a):
        real = self.guard.resolve(a["path"])
        if not os.path.isdir(real):
            raise Refused(f"{real} n'est pas un dossier.")
        entries = []
        for name in sorted(os.listdir(real), key=str.lower):
            full = os.path.join(real, name)
            if self.guard.skip_in_walk(_real(full)) and not os.path.islink(full):
                continue
            try:
                st = os.lstat(full)
            except OSError:
                continue
            kind = "lien" if os.path.islink(full) else "dossier" if os.path.isdir(full) else "fichier"
            when = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            entries.append(f"{kind:8} {_fmt_bytes(st.st_size) if kind == 'fichier' else '':>10}  {when}  {name}")
        shown = entries[:500]
        tail = f"\n… {len(entries) - 500} autres entrées" if len(entries) > 500 else ""
        return Result(True, f"{real} ({len(entries)} entrées)\n" + "\n".join(shown) + tail)

    def do_read_file(self, a):
        real = self.guard.resolve(a["path"])
        if not os.path.isfile(real):
            raise Refused(f"{real} n'est pas un fichier.")
        size = os.path.getsize(real)
        start = a.get("offset", 0)
        with open(real, "rb") as f:
            f.seek(start)
            raw = f.read(a["max_bytes"])
        if b"\x00" in raw[:8192]:
            return Result(True, f"{real} : fichier binaire ({_fmt_bytes(size)}), contenu non affiché.")
        text = raw.decode("utf-8", errors="replace")
        end = start + len(raw)
        note = f"\n…[suite : octets {end} à {size}, avec offset={end}]" if end < size else ""
        span = f", octets {start} à {end}" if start or end < size else ""
        return Result(True, f"{real} ({_fmt_bytes(size)}{span})\n---\n{text}{note}")

    def do_search_files(self, a):
        root = self.guard.resolve(a["path"])
        if not os.path.isdir(root):
            raise Refused(f"{root} n'est pas un dossier.")
        needle = a["query"].lower()
        hits, deadline = [], time.time() + 20
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if not self.guard.skip_in_walk(_real(os.path.join(dirpath, d)))]
            for name in filenames:
                full = os.path.join(dirpath, name)
                if self.guard.skip_in_walk(_real(full)) or os.path.islink(full):
                    continue
                if not a["contents"]:
                    if needle in name.lower():
                        hits.append(full)
                else:
                    try:
                        if os.path.getsize(full) > 2_000_000:
                            continue
                        with open(full, "rb") as f:
                            if b"\x00" in f.read(4096):
                                continue
                        with open(full, "r", encoding="utf-8", errors="replace") as f:
                            for n, line in enumerate(f, 1):
                                if needle in line.lower():
                                    hits.append(f"{full}:{n}: {line.strip()[:200]}")
                                    if len(hits) >= a["max_results"]:
                                        break
                    except OSError:
                        continue
                if len(hits) >= a["max_results"] or time.time() > deadline:
                    break
            if len(hits) >= a["max_results"] or time.time() > deadline:
                break
        more = " (recherche interrompue : limite atteinte)" if len(hits) >= a["max_results"] or time.time() > deadline else ""
        return Result(True, (f"{len(hits)} résultat(s){more}\n" + "\n".join(hits[: a["max_results"]])) if hits else "Aucun résultat.")

    # -- fichiers : écriture -----------------------------------------------------
    def _refuse_program(self, real):
        """Écrire un programme ou un fichier qui se lance tout seul (démarrage, hook git…) = lancer des commandes."""
        if "exec" in self.cfg.enabled:
            return
        parts = [p.lower() for p in re.split(r"[\\/]+", real) if p]
        base = parts[-1] if parts else ""
        hook = ".git" in parts and "hooks" in parts
        if (os.path.splitext(base)[1] in PROGRAM_EXTENSIONS or base in AUTORUN_NAMES or hook
                or any(p in AUTORUN_DIRS for p in parts[:-1])):
            raise Refused("Écrire un programme, un script ou un fichier de démarrage est refusé : cela reviendrait à lancer "
                          "des commandes, ce qui n'est pas activé sur cette machine (python kira_core.py allow exec).")

    def do_write_file(self, a):
        real = self.guard.resolve(a["path"], is_root_ok=False)
        self._refuse_program(real)
        data = a["content"].encode("utf-8")
        exists = os.path.lexists(real)
        if exists and os.path.isdir(real):
            raise Refused(f"{real} est un dossier.")
        if a["mode"] == "create" and exists:
            raise Refused(f"{real} existe déjà (utilise le mode « overwrite » pour le remplacer, une copie ira à la corbeille).")
        os.makedirs(os.path.dirname(real), exist_ok=True)
        if a["mode"] == "append":
            with open(real, "ab") as f:
                f.write(data)
            return Result(True, f"{len(data)} octets ajoutés à {real}.")
        backup = ""
        if exists:
            backup = f" Ancienne version gardée dans la corbeille ({self.trash.put(real, 'remplacé', copy=True)})."
        tmp = os.path.join(os.path.dirname(real), f".kira-tmp-{os.urandom(4).hex()}")
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            if exists:
                try:
                    shutil.copymode(real, tmp)
                except OSError:
                    pass
            os.replace(tmp, real)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return Result(True, f"{len(data)} octets écrits dans {real}.{backup}")

    def do_make_dir(self, a):
        real = self.guard.resolve(a["path"], is_root_ok=False)
        os.makedirs(real, exist_ok=True)
        return Result(True, f"Dossier prêt : {real}")

    def do_move(self, a):
        src = self.guard.resolve(a["src"], is_root_ok=False)
        dst = self.guard.resolve(a["dst"], is_root_ok=False)
        self._refuse_program(dst)
        if not os.path.lexists(src):
            raise Refused(f"{src} n'existe pas.")
        if os.path.lexists(dst):
            raise Refused(f"{dst} existe déjà.")
        if _within(src, dst):
            raise Refused("On ne peut pas déplacer un dossier dans lui-même.")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        return Result(True, f"Déplacé : {src} → {dst}")

    def do_delete(self, a):
        real = self.guard.resolve(a["path"], is_root_ok=False)
        if not os.path.lexists(real):
            raise Refused(f"{real} n'existe pas.")
        entry = self.trash.put(real, "supprimé à la demande de KIRA")
        return Result(True, f"{real} mis à la corbeille de KIRA (identifiant {entry}). "
                            f"Récupérable pendant {TRASH_KEEP_DAYS} jours : python kira_core.py trash restore {entry}")

    # -- lancer ------------------------------------------------------------------
    def _spawn(self, argv, timeout, cwd, shell=False):
        flags = 0
        extra = {}
        if os.name == "nt":
            flags = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        else:
            extra["start_new_session"] = True
        env = dict(os.environ, KIRA_CORE="1")
        proc = subprocess.Popen(argv, shell=shell, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env, creationflags=flags, **extra)
        chunks, size = [], [0]

        def reader():
            for block in iter(lambda: proc.stdout.read(4096), b""):
                if size[0] < 1_000_000:
                    chunks.append(block)
                    size[0] += len(block)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        deadline, why = time.time() + timeout, ""
        while proc.poll() is None:
            if time.time() > deadline:
                why = f"Délai de {timeout} s dépassé : programme arrêté."
            elif self.abort.is_set() or is_paused():
                why = "Arrêté : KIRA a été mis en pause."
            if why:
                self._kill(proc)
                break
            time.sleep(0.15)
        t.join(timeout=3)
        enc = locale.getpreferredencoding(False) or "utf-8"
        text = b"".join(chunks).decode("utf-8" if os.name != "nt" else enc, errors="replace")
        return proc, text, why

    @staticmethod
    def _kill(proc):
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=10)
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, subprocess.SubprocessError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.SubprocessError:
            pass

    def _workdir(self, cwd):
        if cwd:
            real = self.guard.resolve(cwd)
            if not os.path.isdir(real):
                raise Refused(f"{real} n'est pas un dossier.")
            return real
        if not self.guard.roots:
            raise Refused("Aucun dossier autorisé : impossible de choisir où travailler.")
        os.makedirs(self.guard.roots[0], exist_ok=True)
        return self.guard.roots[0]

    def _finish(self, proc, text, why):
        body = _clip(text.strip()) or "(aucune sortie)"
        head = f"{why}\n" if why else ""
        return Result(proc.returncode == 0 and not why, f"{head}Code de sortie : {proc.returncode}\n{body}")

    def do_run_command(self, a):
        proc, text, why = self._spawn(a["command"], a["timeout"], self._workdir(a.get("cwd")), shell=True)
        return self._finish(proc, text, why)

    def do_run_python(self, a):
        work = self._workdir(None)
        fd, path = tempfile.mkstemp(suffix=".py", prefix="kira-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(a["code"])
            proc, text, why = self._spawn([sys.executable, path], a["timeout"], work)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        return self._finish(proc, text, why)

    def do_open(self, a):
        target = a["target"].strip()
        if re.match(r"^https?://", target, re.I):
            where = target
        elif re.match(r"^[A-Za-z][A-Za-z0-9+.\-]+:", target):
            raise Refused("Seuls les liens http(s) et les fichiers des dossiers autorisés peuvent être ouverts.")
        else:
            where = self.guard.resolve(target)
            if not os.path.exists(where):
                raise Refused(f"{where} n'existe pas.")
            if os.path.splitext(where)[1].lower() in EXEC_EXTENSIONS or (os.path.isfile(where) and os.access(where, os.X_OK) and os.name != "nt"):
                raise Refused("Ouvrir un programme est refusé : demande plutôt une commande explicite.")
        if os.name == "nt":
            os.startfile(where)  # noqa: S606 (liste blanche ci-dessus)
        else:
            tool = "open" if sys.platform == "darwin" else "termux-open" if platform_name() == "android" else "xdg-open"
            if not shutil.which(tool):
                raise Refused(f"« {tool} » n'est pas installé sur cette machine.")
            subprocess.Popen([tool, where], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        return Result(True, f"Ouvert : {where}")

    # -- écran, souris, clavier --------------------------------------------------
    def _screenshot_backend(self):
        if _has_module("mss"):
            return "mss"
        if _has_module("PIL") and (os.name == "nt" or sys.platform == "darwin"):
            return "pillow"
        if sys.platform == "darwin" and shutil.which("screencapture"):
            return "screencapture"
        if sys.platform.startswith("linux") and platform_name() != "android":
            for tool in ("grim", "scrot", "maim", "gnome-screenshot", "import"):
                if shutil.which(tool):
                    return tool
        return ""

    def _input_backend(self):
        if _has_module("pyautogui"):
            return "pyautogui"
        if sys.platform.startswith("linux") and platform_name() != "android" and shutil.which("xdotool"):
            return "xdotool"
        return ""

    def _grab(self, backend):
        if backend == "mss":
            import mss
            import mss.tools
            with mss.mss() as sct:
                shot = sct.grab(sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0])
                return mss.tools.to_png(shot.rgb, shot.size)
        if backend == "pillow":
            from PIL import ImageGrab
            buf = io.BytesIO()
            ImageGrab.grab().save(buf, "PNG")
            return buf.getvalue()
        fd, path = tempfile.mkstemp(suffix=".png", prefix="kira-shot-")
        os.close(fd)
        try:
            cmd = {
                "screencapture": ["screencapture", "-x", "-t", "png", path],
                "grim": ["grim", path],
                "scrot": ["scrot", "-o", path],
                "maim": ["maim", path],
                "gnome-screenshot": ["gnome-screenshot", "-f", path],
                "import": ["import", "-window", "root", path],
            }[backend]
            done = subprocess.run(cmd, capture_output=True, timeout=30)
            if done.returncode != 0 or not os.path.getsize(path):
                raise Refused("La capture a échoué : " + done.stderr.decode("utf-8", "replace")[:200])
            with open(path, "rb") as f:
                return f.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def do_screenshot(self, a):
        backend = self._screenshot_backend()
        if not backend:
            raise Refused("Aucun outil de capture sur cette machine (installe « pip install mss pillow », ou scrot/grim sous Linux). "
                          "Android : la capture d'écran n'est pas possible depuis Termux.")
        png = self._grab(backend)
        note = ""
        if _has_module("PIL"):
            from PIL import Image
            img = Image.open(io.BytesIO(png)).convert("RGB")
            full = img.size
            if img.width > SHOT_MAX_WIDTH:
                img = img.resize((SHOT_MAX_WIDTH, round(img.height * SHOT_MAX_WIDTH / img.width)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "PNG", optimize=True)
            out = buf.getvalue()
            if len(out) > SHOT_MAX_BYTES:
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=70)
                out = buf.getvalue()
            sent = img.size
            note = f"Écran {full[0]}×{full[1]} px, image envoyée en {sent[0]}×{sent[1]}."
        else:
            if len(png) > SHOT_MAX_BYTES:
                raise Refused("La capture est trop lourde : installe Pillow (pip install pillow) pour la réduire.")
            out = png
            sent = (struct_png_size(png) or (0, 0))
            note = f"Écran capturé ({sent[0]}×{sent[1]} px)."
        self.last_shot = {"sent": sent}
        return Result(True, note + " Les coordonnées de la souris se lisent sur cette image.",
                      [{"b64": base64.b64encode(out).decode("ascii")}])

    def _screen_size(self):
        if _has_module("pyautogui"):
            import pyautogui
            return tuple(pyautogui.size())
        out = subprocess.run(["xdotool", "getdisplaygeometry"], capture_output=True, text=True, timeout=5).stdout.split()
        if len(out) == 2:
            return int(out[0]), int(out[1])
        raise Refused("Taille de l'écran introuvable.")

    def _throttle_input(self):
        now = time.time()
        if len(self._input_times) == self._input_times.maxlen and now - self._input_times[0] < 60:
            raise Refused("Trop d'actions clavier/souris en une minute (60 maximum) : pause de sécurité.")
        if self._input_times and now - self._input_times[-1] < 0.15:
            time.sleep(0.15)
        self._input_times.append(time.time())

    def _failsafe(self, exc):
        set_paused(True)
        return Result(False, "Arrêt d'urgence : la souris a touché le coin de l'écran. KIRA est en pause sur cette machine "
                             "(python kira_core.py resume pour reprendre).")

    def do_mouse(self, a):
        backend = self._input_backend()
        if not backend:
            raise Refused("Pas d'outil de pilotage ici : installe « pip install pyautogui » (ou xdotool sous Linux).")
        self._throttle_input()
        width, height = self._screen_size()
        x = y = None
        if "x" in a:
            x, y = a["x"], a["y"]
            if a["space"] == "screenshot":
                if not self.last_shot:
                    raise Refused("Prends d'abord une capture d'écran : les coordonnées sont lues dessus.")
                sw, sh = self.last_shot["sent"]
                x, y = x * width / max(sw, 1), y * height / max(sh, 1)
            x, y = int(max(0, min(width - 1, round(x)))), int(max(0, min(height - 1, round(y))))
        act, amount = a["action"], a.get("amount", 0)
        try:
            if backend == "pyautogui":
                import pyautogui
                pyautogui.FAILSAFE = True
                pyautogui.PAUSE = 0.05
                if act == "move":
                    pyautogui.moveTo(x, y, duration=0.15)
                elif act == "click":
                    pyautogui.click(x, y)
                elif act == "double_click":
                    pyautogui.doubleClick(x, y)
                elif act == "right_click":
                    pyautogui.rightClick(x, y)
                else:
                    pyautogui.scroll(amount)
            else:
                run = lambda *args: subprocess.run(["xdotool", *args], capture_output=True, timeout=10, check=True)  # noqa: E731
                if act == "scroll":
                    run("click", "--repeat", str(abs(amount)), "4" if amount > 0 else "5")
                else:
                    run("mousemove", str(x), str(y))
                    if act == "click":
                        run("click", "1")
                    elif act == "double_click":
                        run("click", "--repeat", "2", "--delay", "80", "1")
                    elif act == "right_click":
                        run("click", "3")
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "FailSafeException":
                return self._failsafe(exc)
            raise
        return Result(True, f"{act} fait" + (f" en ({x}, {y}) sur l'écran" if x is not None else "") + ".")

    _XKEYS = {"enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape", "backspace": "BackSpace",
              "delete": "Delete", "del": "Delete", "tab": "Tab", "space": "space", "up": "Up", "down": "Down", "left": "Left",
              "right": "Right", "home": "Home", "end": "End", "pageup": "Prior", "pagedown": "Next", "win": "Super_L",
              "cmd": "Super_L", "super": "Super_L", "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "shift": "shift"}
    _PYKEYS = {"control": "ctrl", "del": "delete", "return": "enter", "esc": "esc", "escape": "esc", "super": "win", "cmd": "command" if sys.platform == "darwin" else "win"}

    def do_keyboard(self, a):
        backend = self._input_backend()
        if not backend:
            raise Refused("Pas d'outil de pilotage ici : installe « pip install pyautogui » (ou xdotool sous Linux).")
        self._throttle_input()
        try:
            if a.get("text"):
                if backend == "pyautogui":
                    import pyautogui
                    if any(ord(c) > 126 for c in a["text"]):
                        raise Refused("pyautogui ne sait taper que l'ASCII : installe xdotool (Linux) ou écris sans accents.")
                    pyautogui.FAILSAFE = True
                    pyautogui.write(a["text"], interval=0.01)
                else:
                    subprocess.run(["xdotool", "type", "--delay", "12", "--", a["text"]], capture_output=True, timeout=60, check=True)
                return Result(True, f"{len(a['text'])} caractères tapés.")
            keys = [k.strip().lower() for k in a["keys"].split("+") if k.strip()]
            if not keys or not all(re.match(r"^[a-z0-9_]{1,12}$", k) for k in keys):
                raise Refused("Touches invalides (exemple : ctrl+s, alt+tab, enter).")
            if backend == "pyautogui":
                import pyautogui
                pyautogui.FAILSAFE = True
                pyautogui.hotkey(*[self._PYKEYS.get(k, k) for k in keys])
            else:
                subprocess.run(["xdotool", "key", "+".join(self._XKEYS.get(k, k) for k in keys)], capture_output=True,
                               timeout=10, check=True)
            return Result(True, f"Touches pressées : {'+'.join(keys)}.")
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "FailSafeException":
                return self._failsafe(exc)
            raise


def struct_png_size(png):
    import struct
    if png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) >= 24:
        return struct.unpack(">II", png[16:24])
    return None


# ------------------------------------------------------------------ liaison avec le serveur
def check_server_url(url):
    url = (url or "").strip().rstrip("/")
    m = re.match(r"^(https?)://([^/:]+)", url, re.I)
    if not m:
        raise Refused("L'adresse du serveur doit commencer par https://")
    local = m.group(2).lower() in ("localhost", "127.0.0.1", "::1", "[::1]")
    if m.group(1).lower() != "https" and not (local or os.environ.get("KIRA_CORE_ALLOW_HTTP") == "1"):
        raise Refused("Connexion non chiffrée refusée : utilise une adresse https:// (le jeton de la machine y circule).")
    return url


def api_post(server, path, body, token=None, timeout=30):
    headers = {"Content-Type": "application/json", "User-Agent": f"kira-core/{VERSION}"}
    if token:
        headers["Authorization"] = "Device " + token
    req = urllib.request.Request(server + path, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read().decode("utf-8")).get("error", "")
        except Exception:  # noqa: BLE001
            message = ""
        raise ApiFail(exc.code, message or f"HTTP {exc.code}")


class Runner:
    def __init__(self, core, log):
        self.core = core
        self.cfg = core.cfg
        self.log = log
        self.stop = threading.Event()
        self.busy = threading.Event()

    def _ping_loop(self):
        while not self.stop.is_set():
            self.stop.wait(25)
            if self.busy.is_set():
                try:
                    reply = api_post(self.cfg.data["server"], "/api/core/ping", {}, self.cfg.data["token"], timeout=15)
                    if reply.get("paused"):
                        self.core.abort.set()
                except Exception:  # noqa: BLE001
                    pass

    def _submit(self, action_id, result):
        body = {"action_id": action_id, "ok": result.ok, "output": result.output, "images": result.images}
        for attempt in range(4):
            try:
                api_post(self.cfg.data["server"], "/api/core/result", body, self.cfg.data["token"], timeout=60)
                return
            except ApiFail as exc:
                if exc.status in (400, 401, 413):
                    self.log.warning("résultat refusé par le serveur : %s", exc)
                    return
            except Exception as exc:  # noqa: BLE001
                self.log.warning("envoi du résultat (essai %d) : %s", attempt + 1, exc)
            time.sleep(2 * (attempt + 1))

    def handle(self, action):
        kind = str(action.get("kind"))
        self.log.info("action %s : %s", action.get("id"), kind)
        self.core.abort.clear()
        self.core.reload()
        self.busy.set()
        try:
            result = self.core.execute(kind, action.get("args") or {})
        finally:
            self.busy.clear()
        self._submit(action["id"], result)
        self.log.info("action %s terminée (%s)", action.get("id"), "ok" if result.ok else "échec")

    def run(self):
        threading.Thread(target=self._ping_loop, daemon=True).start()
        self.core.trash.purge()
        delay = 2
        self.log.info("KIRA Core %s démarré : %s", VERSION, self.cfg.data["server"])
        while not self.stop.is_set():
            self.core.reload()
            try:
                reply = api_post(self.cfg.data["server"], "/api/core/poll", {"info": self.core.info(), "wait": 20},
                                 self.cfg.data["token"], timeout=50)
                delay = 2
            except ApiFail as exc:
                if exc.status == 401:
                    self.log.error("Ce jeton n'est plus valable (machine retirée ?). Arrêt. Refais « pair » si besoin.")
                    return 3
                self.log.warning("serveur : %s", exc)
                self.stop.wait(delay)
                delay = min(60, delay * 2)
                continue
            except Exception as exc:  # noqa: BLE001  (réseau coupé, serveur qui se réveille…)
                self.log.warning("pas de réponse (%s), nouvel essai dans %d s", type(exc).__name__, delay)
                self.stop.wait(delay)
                delay = min(60, delay * 2)
                continue
            if reply.get("paused"):
                self.stop.wait(4)
                continue
            if reply.get("action"):
                self.handle(reply["action"])
        return 0


# ------------------------------------------------------------------ démarrage automatique
def script_target():
    return os.path.join(home_dir(), "kira_core.py")


def render_systemd_unit(python, script):
    return (
        "[Unit]\nDescription=KIRA Core (cœur de KIRA sur cette machine)\nAfter=network-online.target\n\n"
        f"[Service]\nExecStart={shlex.quote(python)} {shlex.quote(script)} run\nRestart=on-failure\nRestartSec=10\n"
        "RestartPreventExitStatus=3\nNoNewPrivileges=true\n\n[Install]\nWantedBy=default.target\n"
    )


def render_launchd_plist(python, script, log_path):
    from xml.sax.saxutils import escape
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0"><dict>\n'
        "<key>Label</key><string>com.kira.core</string>\n<key>ProgramArguments</key><array>"
        f"<string>{escape(python)}</string><string>{escape(script)}</string><string>run</string></array>\n"
        "<key>RunAtLoad</key><true/>\n<key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>\n"
        f"<key>StandardErrorPath</key><string>{escape(log_path)}</string>\n</dict></plist>\n"
    )


def render_termux_boot(python, script):
    return f"#!/data/data/com.termux/files/usr/bin/sh\ntermux-wake-lock\nexec {shlex.quote(python)} {shlex.quote(script)} run\n"


def windows_task_command(python, script):
    py = python
    if py.lower().endswith("python.exe"):
        w = py[:-10] + "pythonw.exe"
        if os.path.exists(w):
            py = w
    return f'"{py}" "{script}" run'


def install_autostart(log=print):
    os.makedirs(home_dir(), exist_ok=True)
    target = script_target()
    me = os.path.abspath(__file__)
    if _norm(me) != _norm(target):
        shutil.copy2(me, target)
    python = sys.executable
    name = platform_name()
    if name == "windows":
        cmd = windows_task_command(python, target)
        subprocess.run(["schtasks", "/Create", "/TN", "KIRA-Core", "/TR", cmd, "/SC", "ONLOGON", "/RL", "LIMITED", "/F"], check=True)
        subprocess.run(["schtasks", "/Run", "/TN", "KIRA-Core"], check=False)
        return "Démarrage automatique installé (Planificateur de tâches : KIRA-Core). Il tourne déjà."
    if name == "mac":
        plist = os.path.expanduser("~/Library/LaunchAgents/com.kira.core.plist")
        os.makedirs(os.path.dirname(plist), exist_ok=True)
        with open(plist, "w", encoding="utf-8") as f:
            f.write(render_launchd_plist(python, target, os.path.join(home_dir(), "core.log")))
        subprocess.run(["launchctl", "unload", plist], capture_output=True)
        subprocess.run(["launchctl", "load", "-w", plist], check=True)
        return "Démarrage automatique installé (launchd : com.kira.core). Il tourne déjà."
    if name == "android":
        boot = os.path.expanduser("~/.termux/boot")
        os.makedirs(boot, exist_ok=True)
        path = os.path.join(boot, "kira-core")
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_termux_boot(python, target))
        os.chmod(path, 0o700)
        return ("Script de démarrage écrit pour Termux:Boot. Installe l'application « Termux:Boot » (F-Droid) et ouvre-la "
                "une fois ; en attendant, lance « python ~/.kira-core/kira_core.py run » dans Termux.")
    if shutil.which("systemctl"):
        unit_dir = os.path.expanduser("~/.config/systemd/user")
        os.makedirs(unit_dir, exist_ok=True)
        with open(os.path.join(unit_dir, "kira-core.service"), "w", encoding="utf-8") as f:
            f.write(render_systemd_unit(python, target))
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "--now", "kira-core.service"], check=True)
        return ("Démarrage automatique installé (systemd, service utilisateur kira-core). "
                "Pour qu'il tourne aussi sans session ouverte : sudo loginctl enable-linger $USER")
    return f"systemd introuvable. Lance au démarrage : {python} {target} run"


def uninstall_autostart():
    name = platform_name()
    if name == "windows":
        subprocess.run(["schtasks", "/End", "/TN", "KIRA-Core"], capture_output=True)
        subprocess.run(["schtasks", "/Delete", "/TN", "KIRA-Core", "/F"], capture_output=True)
    elif name == "mac":
        plist = os.path.expanduser("~/Library/LaunchAgents/com.kira.core.plist")
        subprocess.run(["launchctl", "unload", plist], capture_output=True)
        if os.path.exists(plist):
            os.remove(plist)
    elif name == "android":
        path = os.path.expanduser("~/.termux/boot/kira-core")
        if os.path.exists(path):
            os.remove(path)
    elif shutil.which("systemctl"):
        subprocess.run(["systemctl", "--user", "disable", "--now", "kira-core.service"], capture_output=True)
        path = os.path.expanduser("~/.config/systemd/user/kira-core.service")
        if os.path.exists(path):
            os.remove(path)
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    return "Démarrage automatique retiré."


# ------------------------------------------------------------------ ligne de commande
def make_logger(console=True):
    log = logging.getLogger("kira-core")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
        try:
            os.makedirs(home_dir(), exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(os.path.join(home_dir(), "core.log"), maxBytes=200_000, backupCount=2,
                                                      encoding="utf-8")
            fh.setFormatter(fmt)
            log.addHandler(fh)
        except OSError:
            pass
        if console and sys.stderr is not None:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            log.addHandler(sh)
    return log


def cmd_pair(args):
    cfg = Config()
    if cfg.paired and not args.force:
        raise Refused("Cette machine est déjà appairée. Pour recommencer : « unpair », ou ajoute --force.")
    server = check_server_url(args.server)
    roots = [os.path.abspath(os.path.expanduser(r)) for r in (args.root or [])] or [default_root()]
    for r in roots:
        os.makedirs(r, exist_ok=True)
    cfg.data.update({"server": server, "token": "", "roots": roots,
                     "enabled": list(DEFAULT_ENABLED) + (list(SENSITIVE_CATEGORIES) if args.full else [])})
    core = Core(cfg)
    name = args.name or socket.gethostname()
    reply = api_post(server, "/api/core/pair", {"code": args.code, "name": name, "platform": platform_name(),
                                                 "info": core.info(), "version": VERSION})
    cfg.data.update({"token": reply["token"], "device_id": reply["device_id"], "name": reply["name"]})
    cfg.save()
    lines = [f"Appairé : « {reply['name']} » (n°{reply['device_id']}).",
             f"Pouvoirs actifs ici : {', '.join(CATEGORIES[c] for c in cfg.enabled)}.",
             f"Dossiers autorisés : {', '.join(roots)}."]
    if not args.full:
        lines.append("Lancer des commandes, voir l'écran et piloter souris/clavier restent coupés. Pour les activer ici : "
                     "python kira_core.py allow exec | screen | input")
    lines.append("Étape suivante : « python kira_core.py install » (démarrage automatique) ou « python kira_core.py run ».")
    return "\n".join(lines)


def cmd_status(args):
    cfg = Config()
    if not cfg.paired:
        return "Pas encore appairée. Utilise « pair » avec le code affiché dans l'application (Plus → Mes appareils)."
    core = Core(cfg)
    av = core.available()
    lines = [
        f"KIRA Core {VERSION} · {cfg.data.get('name')} (n°{cfg.data.get('device_id')}) · {platform_name()}",
        f"Serveur : {cfg.data['server']}",
        f"Pause locale : {'OUI' if is_paused() else 'non'}",
        "Pouvoirs : " + ", ".join(f"{CATEGORIES[c]} [{'actif' if c in cfg.enabled else 'coupé'}]" for c in CATEGORIES),
        f"Dossiers : {', '.join(cfg.roots) or 'aucun'}",
        f"Capture d'écran : {av['screenshot'] or 'indisponible'} · Souris/clavier : {av['input'] or 'indisponible'}",
        f"Fichiers de KIRA Core : {home_dir()}",
    ]
    return "\n".join(lines)


def cmd_toggle(args, on):
    cfg = Config()
    if args.category not in CATEGORIES:
        raise Refused(f"Catégorie inconnue. Choix : {', '.join(CATEGORIES)}.")
    enabled = set(cfg.enabled)
    (enabled.add if on else enabled.discard)(args.category)
    cfg.data["enabled"] = [c for c in CATEGORIES if c in enabled]
    cfg.save()
    note = " Le serveur le saura à la prochaine liaison (quelques secondes)." if cfg.paired else ""
    return f"« {CATEGORIES[args.category]} » : {'activé' if on else 'coupé'} sur cette machine.{note}"


def cmd_roots(args):
    cfg = Config()
    roots = cfg.roots
    if args.op == "list":
        return "\n".join(roots) or "Aucun dossier autorisé."
    path = os.path.abspath(os.path.expanduser(args.path or ""))
    if not args.path:
        raise Refused("Indique un dossier.")
    if args.op == "add":
        if path == os.path.abspath(os.sep) or _norm(path) == _norm(os.path.expanduser("~")):
            sys.stderr.write("Attention : ce dossier est très large. Les fichiers sensibles restent interdits, mais préfère un sous-dossier.\n")
        os.makedirs(path, exist_ok=True)
        if path not in roots:
            roots.append(path)
    else:
        roots = [r for r in roots if _norm(r) != _norm(path)]
    cfg.data["roots"] = roots
    cfg.save()
    return "Dossiers autorisés : " + (", ".join(roots) or "aucun")


def cmd_trash(args):
    trash = Trash()
    if args.op == "list":
        entries = trash.entries()
        return "\n".join(f"{e['id']}  {e['when']}  {e['original']}  ({e['reason']})" for e in entries) or "La corbeille est vide."
    return "Restauré : " + trash.restore(args.id or "")


def cmd_unpair(args):
    cfg = Config()
    cfg.data.update({"server": "", "token": "", "device_id": 0})
    cfg.save()
    return "Cette machine a oublié le serveur. Dans l'application : Plus → Mes appareils → Retirer, pour couper aussi côté serveur."


def cmd_run(args):
    cfg = Config()
    if not cfg.paired:
        raise Refused("Pas encore appairée : lance d'abord « pair ».")
    check_server_url(cfg.data["server"])
    log = make_logger()
    runner = Runner(Core(cfg, log), log)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: runner.stop.set())
        except (ValueError, OSError):
            pass
    return runner.run()


def build_parser():
    p = argparse.ArgumentParser(prog="kira_core.py", description="KIRA Core : le cœur de KIRA sur cette machine.")
    p.add_argument("--version", action="version", version=f"KIRA Core {VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pair", help="appairer cette machine")
    s.add_argument("--server", required=True)
    s.add_argument("--code", required=True)
    s.add_argument("--name")
    s.add_argument("--full", action="store_true", help="activer aussi commandes, écran, souris/clavier")
    s.add_argument("--root", action="append", help="dossier autorisé (répétable ; défaut ~/KIRA)")
    s.add_argument("--force", action="store_true")
    sub.add_parser("run", help="tourner en continu")
    sub.add_parser("install", help="démarrage automatique")
    sub.add_parser("uninstall", help="retirer le démarrage automatique")
    sub.add_parser("status", help="état local")
    for name in ("allow", "deny"):
        s = sub.add_parser(name, help=f"{name} un pouvoir sur cette machine")
        s.add_argument("category", choices=list(CATEGORIES))
    s = sub.add_parser("roots", help="dossiers autorisés")
    s.add_argument("op", choices=["list", "add", "remove"])
    s.add_argument("path", nargs="?")
    sub.add_parser("pause", help="pause locale immédiate")
    sub.add_parser("resume", help="reprendre")
    s = sub.add_parser("trash", help="corbeille de KIRA")
    s.add_argument("op", choices=["list", "restore"])
    s.add_argument("id", nargs="?")
    sub.add_parser("unpair", help="oublier le serveur")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "run":
            return cmd_run(args)
        out = {
            "pair": cmd_pair, "status": cmd_status, "roots": cmd_roots, "trash": cmd_trash, "unpair": cmd_unpair,
            "allow": lambda a: cmd_toggle(a, True), "deny": lambda a: cmd_toggle(a, False),
            "install": lambda a: install_autostart(), "uninstall": lambda a: uninstall_autostart(),
            "pause": lambda a: (set_paused(True), "KIRA est en pause sur cette machine. « resume » pour reprendre.")[1],
            "resume": lambda a: (set_paused(False), "KIRA reprend sur cette machine.")[1],
        }[args.cmd](args)
        print(out)
        return 0
    except Refused as exc:
        print(f"Non : {exc}", file=sys.stderr)
        return 2
    except ApiFail as exc:
        print(f"Le serveur répond : {exc}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, OSError) as exc:
        print(f"Impossible de joindre le serveur ({exc}). S'il dormait, réessaie dans une minute.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
