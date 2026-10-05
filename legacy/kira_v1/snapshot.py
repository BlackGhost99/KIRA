import datetime
import json
import os
import shutil

from audit import log

SNAPSHOTS_DIR = "snapshots"
MANIFEST_FILE = "manifest.json"


def ensure_snapshots_dir():
    if not os.path.exists(SNAPSHOTS_DIR):
        os.makedirs(SNAPSHOTS_DIR)


def create_snapshot(description: str = ""):
    ensure_snapshots_dir()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_id = f"snapshot_{timestamp}"
    snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_id)

    # Copie complète du projet (sauf le dossier snapshots pour éviter la récursion)
    shutil.copytree(
        ".",
        snapshot_path,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(SNAPSHOTS_DIR),
    )

    # Création du manifest
    manifest = {
        "id": snapshot_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "description": description,
        "proprietaire": "Brice",
    }

    with open(os.path.join(snapshot_path, MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log("snapshot_create", {"snapshot_id": snapshot_id, "description": description})
    return snapshot_id


def list_snapshots():
    ensure_snapshots_dir()
    snapshots = []
    for item in os.listdir(SNAPSHOTS_DIR):
        item_path = os.path.join(SNAPSHOTS_DIR, item)
        manifest_path = os.path.join(item_path, MANIFEST_FILE)
        if os.path.isdir(item_path) and os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            snapshots.append(manifest)
    # Tri par date décroissante
    snapshots.sort(key=lambda x: x["timestamp"], reverse=True)
    return snapshots


def rollback_to(snapshot_id: str):
    snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_id)
    manifest_path = os.path.join(snapshot_path, MANIFEST_FILE)

    if not os.path.exists(snapshot_path) or not os.path.exists(manifest_path):
        log("snapshot_rollback_failed", {"snapshot_id": snapshot_id, "reason": "not_found"})
        return "Snapshot introuvable."

    # Confirmation de sécurité (on ne supprime pas l'état actuel, on le sauvegarde d'abord)
    backup_id = create_snapshot(description="AUTO_BACKUP avant rollback")

    # Suppression de l'état actuel (sauf snapshots)
    for item in os.listdir("."):
        if item != SNAPSHOTS_DIR and item != ".git":  # préserve snapshots et éventuel git
            item_path = os.path.abspath(item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

    # Restoration du snapshot
    for item in os.listdir(snapshot_path):
        if item != MANIFEST_FILE:
            shutil.copytree(
                os.path.join(snapshot_path, item),
                os.path.join(".", item),
                dirs_exist_ok=True,
            )
        else:
            # Copie simple du manifest dans le snapshot (pour traçabilité)
            shutil.copy(manifest_path, os.path.join(snapshot_path, item))

    log(
        "snapshot_rollback_success",
        {
            "from_snapshot": snapshot_id,
            "auto_backup": backup_id,
        },
    )
    return (
        f"Rollback effectué vers {snapshot_id}. "
        f"Un backup automatique a été créé : {backup_id}."
    )
