"""Exécution de code Python dans un processus séparé, limité et sans secrets.

Garde-fous : processus isolé (python -I), environnement vide (aucune clé d'API),
délai maximal, limites de mémoire / CPU / fichiers / processus posées par le
processus lui-même, dossier temporaire jetable, et — quand le serveur tourne en
root, comme dans Docker — passage à l'utilisateur « nobody » pour que le code ne
puisse ni lire les secrets du serveur ni écrire ailleurs que dans son dossier.

Limite assumée : le réseau n'est pas coupé. Le code n'est lancé que sur demande
de Brice (authentifié) et peut être produit par l'IA : à garder en tête.

FICHIER PROTÉGÉ : l'évolution ne peut pas le modifier.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile

from . import config

MAX_OUTPUT_CHARS = 6000
MAX_FILES = 6
MAX_FILE_BYTES = 3_000_000
NOBODY = 65534

# Palette du carnet de nuit de l'interface : les graphiques s'y fondent.
PRELUDE = r'''
import os, sys, warnings
warnings.filterwarnings("ignore")
try:
    import resource
    _cpu = __CPU__
    resource.setrlimit(resource.RLIMIT_CPU, (_cpu, _cpu))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (40 * 1024**2, 40 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
except Exception:
    pass
OUT_DIR = __OUT__
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#181C38", "axes.facecolor": "#181C38", "savefig.facecolor": "#181C38",
        "axes.edgecolor": "#6A72A8", "axes.labelcolor": "#ECE9E1", "text.color": "#ECE9E1",
        "xtick.color": "#C9CBE0", "ytick.color": "#C9CBE0", "grid.color": "#2F3668",
        "axes.grid": True, "grid.alpha": 0.8, "legend.facecolor": "#232850",
        "legend.edgecolor": "#2F3668", "figure.figsize": (7, 4.2), "font.size": 11,
        "axes.prop_cycle": matplotlib.cycler(color=["#F2B84B", "#62D2B5", "#8AA2FF", "#F0766F", "#C79BFF"]),
    })
    _kira_n = [0]
    def _kira_save():
        for num in plt.get_fignums():
            _kira_n[0] += 1
            plt.figure(num).savefig(os.path.join(OUT_DIR, "figure_%d.png" % _kira_n[0]), dpi=130, bbox_inches="tight")
        plt.close("all")
    plt.show = lambda *a, **k: _kira_save()
except Exception:
    def _kira_save():
        pass
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "user.py"), encoding="utf-8").read()
exec(compile(_src, "user.py", "exec"), {"__name__": "__main__"})
_kira_save()
'''


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} caractères coupés]"


def run_python(code: str, timeout: int | None = None) -> dict:
    """Exécute ``code`` ; renvoie stdout, stderr, fichiers produits et l'état."""
    timeout = timeout or config.settings.sandbox_timeout
    tmp = tempfile.mkdtemp(prefix="kira_")
    out_dir = os.path.join(tmp, "out")
    os.makedirs(out_dir)
    os.chmod(tmp, 0o777)
    os.chmod(out_dir, 0o777)
    try:
        with open(os.path.join(tmp, "user.py"), "w", encoding="utf-8") as fh:
            fh.write(code)
        runner = PRELUDE.replace("__OUT__", repr(out_dir)).replace("__CPU__", str(timeout + 5))
        with open(os.path.join(tmp, "main.py"), "w", encoding="utf-8") as fh:
            fh.write(runner)
        for name in ("user.py", "main.py"):
            os.chmod(os.path.join(tmp, name), 0o644)

        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": tmp,
            "TMPDIR": tmp,
            "MPLCONFIGDIR": tmp,
            "MPLBACKEND": "Agg",
            "LANG": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        kwargs: dict = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                kwargs.update(user=NOBODY, group=NOBODY, extra_groups=[])
        proc = subprocess.Popen(
            [sys.executable, "-I", os.path.join(tmp, "main.py")],
            cwd=tmp,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )
        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, AttributeError):
                proc.kill()
            out, err = proc.communicate()

        files = []
        for name in sorted(os.listdir(out_dir))[:MAX_FILES]:
            path = os.path.join(out_dir, name)
            if name.lower().endswith(".png") and os.path.getsize(path) <= MAX_FILE_BYTES:
                with open(path, "rb") as fh:
                    files.append((name, fh.read()))
        return {
            "ok": (proc.returncode == 0) and not timed_out,
            "timed_out": timed_out,
            "returncode": proc.returncode,
            "stdout": _clip(out.decode("utf-8", "replace")),
            "stderr": _clip(err.decode("utf-8", "replace"), 3000),
            "files": files,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
