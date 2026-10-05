"""Évolution de KIRA : il lit son VRAI code, propose une modification, la vérifie, puis attend Brice.

Chaîne complète, sans jamais rien appliquer seul :
  1. KIRA reçoit son code source réel, ses statistiques d'usage et les « pouces vers le bas » de Brice.
  2. Il propose UN changement précis (remplacements exacts ou nouveaux fichiers).
  3. Le changement est appliqué dans une copie, compilé, et la suite de tests est lancée.
  4. Brice relit le diff sur son téléphone et approuve ou rejette.
  5. À l'approbation seulement, KIRA ouvre une pull request GitHub. Brice la fusionne (ou non) ;
     Render redéploie alors la nouvelle version. Aucune fusion automatique.

FICHIER PROTÉGÉ : l'évolution ne peut pas se modifier elle-même.
"""
from __future__ import annotations

import base64
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from urllib.parse import quote

from . import audit, config, net
from .db import get_db, jdump, jload, now_iso
from .llm import get_router
from .principles import ALLOWED_PREFIXES, FORBIDDEN_PREFIXES, PROTECTED_PATHS
from .textutil import clip, extract_json

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_EXT = {".py", ".js", ".css", ".html", ".md", ".json"}
MAX_FILES = 6
MAX_NEW_FILE_CHARS = 80_000
SOURCE_BUDGET_CHARS = 260_000


class EvolutionError(Exception):
    pass


# -- règles de périmètre ---------------------------------------------------
def norm_path(path: str) -> str:
    return str(path or "").strip().replace("\\", "/").lstrip("/")


def check_path(path: str) -> str | None:
    """Renvoie un message d'erreur si le fichier ne peut pas être modifié, sinon None."""
    if not path or ".." in path.split("/"):
        return f"chemin invalide : « {path} »"
    if path in PROTECTED_PATHS:
        return f"{path} est protégé (principes, sécurité ou vérification) : KIRA ne peut pas le modifier"
    if not path.startswith(ALLOWED_PREFIXES):
        return f"{path} est hors du périmètre autorisé ({', '.join(ALLOWED_PREFIXES)})"
    if path.startswith(FORBIDDEN_PREFIXES):
        return f"{path} est hors de portée"
    if Path(path).suffix not in ALLOWED_EXT:
        return f"{path} : type de fichier non autorisé"
    return None


def read_local(path: str) -> str | None:
    target = (ROOT / path).resolve()
    if ROOT not in target.parents or not target.is_file():
        return None
    return target.read_text(encoding="utf-8")


def apply_changes(changes, read) -> tuple[dict[str, str], list[str]]:
    """Applique les changements en mémoire. Renvoie (fichiers_nouveau_contenu, erreurs)."""
    errors: list[str] = []
    files: dict[str, str] = {}
    if not isinstance(changes, list) or not changes:
        return {}, ["aucun changement fourni"]
    for i, ch in enumerate(changes, 1):
        if not isinstance(ch, dict):
            errors.append(f"changement {i} : format invalide")
            continue
        path = norm_path(ch.get("path", ""))
        problem = check_path(path)
        if problem:
            errors.append(f"changement {i} : {problem}")
            continue
        action = ch.get("action")
        if action == "edit":
            current = files[path] if path in files else read(path)
            if current is None:
                errors.append(f"changement {i} : {path} n'existe pas")
                continue
            find, replace = ch.get("find"), ch.get("replace")
            if not isinstance(find, str) or not find or not isinstance(replace, str):
                errors.append(f"changement {i} : « find » et « replace » doivent être du texte (find non vide)")
                continue
            n = current.count(find)
            if n != 1:
                errors.append(
                    f"changement {i} : l'extrait « {clip(find, 70)} » apparaît {n} fois dans {path} "
                    "(il doit apparaître exactement une fois, copié tel quel)"
                )
                continue
            files[path] = current.replace(find, replace, 1)
        elif action == "create":
            content = ch.get("content")
            if path in files or read(path) is not None:
                errors.append(f"changement {i} : {path} existe déjà (utilise action « edit »)")
            elif not isinstance(content, str) or not content.strip():
                errors.append(f"changement {i} : contenu manquant pour {path}")
            elif len(content) > MAX_NEW_FILE_CHARS:
                errors.append(f"changement {i} : {path} est trop long")
            else:
                files[path] = content
        else:
            errors.append(f"changement {i} : action inconnue « {action} » (edit ou create)")
    if len(files) > MAX_FILES:
        errors.append(f"trop de fichiers modifiés ({len(files)} > {MAX_FILES})")
    for path, content in files.items():
        if path.endswith(".py"):
            try:
                compile(content, path, "exec")
            except SyntaxError as exc:
                errors.append(f"{path} : erreur de syntaxe ligne {exc.lineno} ({exc.msg})")
    return files, errors


def make_diff(files: dict[str, str], read) -> str:
    out: list[str] = []
    for path, new in sorted(files.items()):
        old = read(path) or ""
        out.extend(
            difflib.unified_diff(
                old.splitlines(), new.splitlines(), fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=3
            )
        )
    return "\n".join(out)


# -- vérification par les tests -------------------------------------------
def run_tests(files: dict[str, str]) -> tuple[bool | None, str]:
    """Lance la suite de tests sur une COPIE du code avec les changements appliqués."""
    if not config.settings.evolution_run_tests:
        return None, "Tests désactivés (EVOLUTION_RUN_TESTS=0)."
    if os.environ.get("KIRA_NESTED_TESTS"):
        return None, "Tests non relancés (déjà dans une exécution de tests)."
    if not (ROOT / "tests").is_dir():
        return None, "Aucun dossier tests/ : rien à exécuter."
    tmp = Path(tempfile.mkdtemp(prefix="kira_evo_"))
    try:
        for folder in ("app", "tests", "web", "core"):  # core/ : l'agent du cœur, que ses tests exécutent
            if (ROOT / folder).is_dir():
                shutil.copytree(
                    ROOT / folder, tmp / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "vendor", "*.db")
                )
        for path, content in files.items():
            target = tmp / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(tmp),
            "LANG": "C.UTF-8",
            "KIRA_NESTED_TESTS": "1",
            "DATABASE_URL": f"sqlite:///{tmp}/test.db",
        }
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
            cwd=tmp, env=env, capture_output=True, text=True, timeout=240,
        )
        return proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    except subprocess.TimeoutExpired:
        return False, "Délai dépassé (240 s) pendant les tests."
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -- contexte donné à KIRA -------------------------------------------------
def source_snapshot(budget: int = SOURCE_BUDGET_CHARS) -> str:
    blocks: list[str] = []
    total = 0
    # le code vient avant les tests : si le budget de contexte manque, ce sont les tests qui sont omis
    for path in sorted(ROOT.rglob("*"), key=lambda p: (p.relative_to(ROOT).as_posix().startswith("tests/"), p.as_posix())):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if not rel.startswith(ALLOWED_PREFIXES) or rel.startswith(FORBIDDEN_PREFIXES) or "__pycache__" in rel:
            continue
        if path.suffix not in ALLOWED_EXT:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        block = f"=== {rel} ===\n{text}\n"
        if total + len(block) > budget:
            blocks.append(f"=== {rel} === (omis : budget de contexte atteint)\n")
            continue
        blocks.append(block)
        total += len(block)
    return "\n".join(blocks)


def usage_report() -> str:
    db = get_db()
    counts = audit.counts_since(14)
    lines = ["Journal des 14 derniers jours (action : nombre) :"]
    lines += [f"- {k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:20]] or ["- (vide)"]
    errors = db.q(
        "SELECT action, details FROM audit WHERE action IN ('llm_error','tool_error','chat_error','job_error') "
        "ORDER BY id DESC LIMIT 8"
    )
    if errors:
        lines.append("\nDernières erreurs :")
        lines += [f"- {e['action']}: {clip(e['details'], 260)}" for e in errors]
    bad = db.q(
        "SELECT m.content AS answer, m.feedback_note AS note, "
        "(SELECT u.content FROM messages u WHERE u.conversation_id = m.conversation_id AND u.id < m.id "
        " AND u.role = 'user' ORDER BY u.id DESC LIMIT 1) AS question "
        "FROM messages m WHERE m.feedback = -1 ORDER BY m.id DESC LIMIT 8"
    )
    if bad:
        lines.append("\nRéponses jugées mauvaises par Brice (pouce vers le bas) :")
        for b in bad:
            lines.append(f"- Question : {clip(b['question'] or '', 200)}\n  Réponse : {clip(b['answer'], 300)}"
                         + (f"\n  Remarque de Brice : {clip(b['note'], 200)}" if b["note"] else ""))
    previous = db.q("SELECT title, status FROM proposals ORDER BY id DESC LIMIT 10")
    if previous:
        lines.append("\nPropositions déjà faites (ne les répète pas) :")
        lines += [f"- {p['title']} [{p['status']}]" for p in previous]
    return "\n".join(lines)


SYSTEM = """Tu es le module d'évolution de KIRA, l'assistant personnel de {owner}. Tu améliores le code de KIRA en proposant \
UNE modification précise, petite et vérifiable. Tu réponds uniquement par un objet JSON valide.

Règles :
- Appuie-toi sur le code réel fourni (jamais sur des fichiers ou des fonctions imaginés).
- Chaque changement est soit {{"path": "...", "action": "edit", "find": "...", "replace": "..."}} où « find » est un \
extrait EXACT et UNIQUE du fichier actuel, copié tel quel avec ses espaces et retours à la ligne, soit \
{{"path": "...", "action": "create", "content": "..."}} pour un nouveau fichier.
- Modifiable : app/, web/ (sauf web/vendor/) et tests/. Protégés, jamais modifiables : {protected}.
- Quand tu changes un comportement, ajoute ou adapte un test dans tests/ (module unittest).
- Aucune nouvelle dépendance, aucun secret écrit en clair, aucun contournement des principes de KIRA.
- Reste petit : au plus 3 fichiers. Une amélioration qui compte vaut mieux que dix retouches.
- Si rien ne mérite d'être changé, réponds {{"proposal": null, "reason": "..."}}.

Format de réponse : {{"proposal": {{"title": "titre court", "rationale": "pourquoi, en 3 à 6 phrases, avec les \
données qui le justifient", "risks": "ce qui pourrait mal tourner", "changes": [ ... ]}}}}"""


def build_prompt(goal: str) -> str:
    task = (
        f"Objectif demandé par {config.settings.owner_name} : {goal.strip()}"
        if goal.strip()
        else "Aucun objectif imposé : choisis l'amélioration la plus utile d'après les données d'usage ci-dessous "
        "(erreurs, réponses mal notées, limites visibles dans le code)."
    )
    return f"{task}\n\n## Données d'usage\n{usage_report()}\n\n## Code source actuel\n{source_snapshot()}"


# -- proposition -----------------------------------------------------------
def propose(goal: str = "", trigger: str = "manual") -> dict:
    router = get_router()
    system = SYSTEM.format(owner=config.settings.owner_name, protected=", ".join(sorted(PROTECTED_PATHS)))
    messages = [{"role": "user", "content": build_prompt(goal)}]
    candidate: dict | None = None
    last_problem = ""
    for _ in range(3):
        res = router.complete([system], messages, None, tier="deep", max_tokens=12000, purpose="evolution")
        data = extract_json(res.text)
        problem = ""
        if not isinstance(data, dict):
            problem = "Ta réponse n'est pas un objet JSON valide. Réponds uniquement par le JSON demandé."
        elif data.get("proposal") is None:
            audit.log("evolution_none", {"reason": clip(str(data.get("reason", "")), 300)})
            return {"status": "none", "reason": str(data.get("reason", ""))}
        elif not isinstance(data["proposal"], dict):
            problem = "Le champ « proposal » doit être un objet."
        else:
            prop = data["proposal"]
            files, errors = apply_changes(prop.get("changes"), read_local)
            if errors:
                problem = "Ces changements ne peuvent pas être appliqués :\n- " + "\n- ".join(errors)
            else:
                ok, output = run_tests(files)
                candidate = {"prop": prop, "files": files, "tests_ok": ok, "tests_output": output}
                if ok is not False:
                    break
                problem = "La suite de tests échoue avec ce changement :\n" + output + "\nCorrige ta proposition."
        last_problem = problem
        messages += [{"role": "assistant", "content": res.text}, {"role": "user", "content": problem}]
    if candidate is None:
        raise EvolutionError("KIRA n'a pas réussi à produire une proposition valide. " + clip(last_problem, 400))
    prop, files = candidate["prop"], candidate["files"]
    ts = now_iso()
    pid = get_db().insert(
        "proposals",
        {
            "title": clip(str(prop.get("title") or "Amélioration"), 160),
            "rationale": clip(str(prop.get("rationale") or "") + ("\n\nRisques : " + str(prop["risks"]) if prop.get("risks") else ""), 4000),
            "trigger_kind": trigger,
            "changes": jdump(prop["changes"]),
            "diff": make_diff(files, read_local),
            "tests_ok": None if candidate["tests_ok"] is None else int(candidate["tests_ok"]),
            "tests_output": candidate["tests_output"],
            "status": "draft",
            "created_at": ts,
            "updated_at": ts,
        },
    )
    audit.log("evolution_proposed", {"proposal": pid, "files": sorted(files), "tests_ok": candidate["tests_ok"]})
    return {"status": "created", "id": pid}


# -- consultation ----------------------------------------------------------
def _public(row: dict) -> dict:
    changes = jload(row.get("changes"), [])
    return {
        "id": row["id"],
        "title": row["title"],
        "rationale": row["rationale"],
        "trigger": row["trigger_kind"],
        "diff": row["diff"],
        "tests_ok": None if row["tests_ok"] is None else bool(row["tests_ok"]),
        "tests_output": row["tests_output"] or "",
        "status": row["status"],
        "branch": row["branch"],
        "pr_url": row["pr_url"],
        "error": row["error"],
        "files": sorted({norm_path(c.get("path", "")) for c in changes if isinstance(c, dict)}),
        "created_at": row["created_at"],
    }


def list_proposals(limit: int = 30) -> list[dict]:
    return [_public(r) for r in get_db().q("SELECT * FROM proposals ORDER BY id DESC LIMIT ?", [limit])]


def get_proposal(pid: int) -> dict | None:
    row = get_db().q1("SELECT * FROM proposals WHERE id = ?", [pid])
    return _public(row) if row else None


def reject(pid: int) -> dict | None:
    get_db().update("proposals", pid, {"status": "rejected", "updated_at": now_iso()})
    audit.log("evolution_rejected", {"proposal": pid}, actor="owner")
    return get_proposal(pid)


# -- ouverture de la pull request -----------------------------------------
def _gh(method: str, path: str, **kwargs):
    token = config.settings.github_token
    return net.session().request(
        method,
        "https://api.github.com" + path,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30,
        **kwargs,
    )


def _slug(text: str) -> str:
    folded = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", folded).strip("-")[:40] or "evolution"


def open_pr(row: dict) -> dict:
    s = config.settings
    if not s.github_token:
        raise EvolutionError(
            "GITHUB_TOKEN n'est pas configuré : impossible d'ouvrir la pull request. Ajoutez un jeton GitHub "
            "(droit « Contents » et « Pull requests » sur le dépôt) dans les variables d'environnement."
        )
    repo, base = s.github_repo, s.github_base_branch
    shas: dict[str, str] = {}

    def read_remote(path: str) -> str | None:
        r = _gh("GET", f"/repos/{repo}/contents/{quote(path)}", params={"ref": base})
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise EvolutionError(f"GitHub : lecture de {path} impossible (HTTP {r.status_code}).")
        data = r.json()
        shas[path] = data["sha"]
        return base64.b64decode(data["content"]).decode("utf-8")

    files, errors = apply_changes(jload(row["changes"], []), read_remote)
    if errors:
        raise EvolutionError("Le dépôt a changé depuis la proposition : " + " ; ".join(errors))
    ref = _gh("GET", f"/repos/{repo}/git/ref/heads/{base}")
    if ref.status_code != 200:
        raise EvolutionError(f"GitHub : branche « {base} » introuvable (HTTP {ref.status_code}).")
    branch = f"kira/evolution-{row['id']}-{_slug(row['title'])}"
    made = _gh("POST", f"/repos/{repo}/git/refs", json={"ref": f"refs/heads/{branch}", "sha": ref.json()["object"]["sha"]})
    if made.status_code not in (200, 201):
        raise EvolutionError(f"GitHub : création de la branche impossible (HTTP {made.status_code} : {made.text[:200]}).")
    for path, content in sorted(files.items()):
        body = {
            "message": f"KIRA : {row['title']}\n\nProposition d'évolution #{row['id']} (fichier {path}).",
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if path in shas:
            body["sha"] = shas[path]
        put = _gh("PUT", f"/repos/{repo}/contents/{quote(path)}", json=body)
        if put.status_code not in (200, 201):
            raise EvolutionError(f"GitHub : écriture de {path} impossible (HTTP {put.status_code} : {put.text[:200]}).")
    tests = {1: "réussis", 0: "ÉCHOUÉS"}.get(row["tests_ok"], "non exécutés")
    pr = _gh(
        "POST",
        f"/repos/{repo}/pulls",
        json={
            "title": f"KIRA : {row['title']}",
            "head": branch,
            "base": base,
            "body": f"{row['rationale']}\n\n---\nTests lancés par KIRA avant l'ouverture : {tests}.\n"
            "Proposition générée par KIRA, validée par son propriétaire. Rien n'est déployé tant que cette "
            "pull request n'est pas fusionnée.",
        },
    )
    if pr.status_code not in (200, 201):
        raise EvolutionError(f"GitHub : ouverture de la pull request impossible (HTTP {pr.status_code} : {pr.text[:200]}).")
    return {"branch": branch, "pr_url": pr.json()["html_url"]}


def approve(pid: int) -> dict:
    row = get_db().q1("SELECT * FROM proposals WHERE id = ?", [pid])
    if not row:
        raise EvolutionError("Proposition introuvable.")
    if row["status"] != "draft":
        raise EvolutionError(f"Cette proposition est déjà « {row['status']} ».")
    if row["tests_ok"] == 0:
        raise EvolutionError("Les tests échouent avec cette proposition : elle ne peut pas être approuvée.")
    audit.log("evolution_approved", {"proposal": pid}, actor="owner")
    try:
        result = open_pr(row)
    except EvolutionError as exc:
        get_db().update("proposals", pid, {"error": str(exc)[:500], "updated_at": now_iso()})
        raise
    get_db().update(
        "proposals",
        pid,
        {"status": "pr_opened", "branch": result["branch"], "pr_url": result["pr_url"], "error": None, "updated_at": now_iso()},
    )
    audit.log("evolution_pr_opened", {"proposal": pid, **result}, actor="owner")
    return get_proposal(pid)
