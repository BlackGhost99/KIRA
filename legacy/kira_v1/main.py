
import datetime
import json
import os

import config

from audit import log
from core import process_input
from evolution import EVOLUTION_DIR, analyze_recent, propose_patch_from_analysis
from observation import (
    OBSERVATION_SOURCES_FILE,
    load_sources,
    observe_all,
    get_last_veille_time,
    reject_last,
    validate_pending,
)
from snapshot import create_snapshot, list_snapshots, rollback_to

print("KIRA – Système d’intelligence personnelle souveraine")
print("Propriétaire : Brice")
print("Tapez 'quit' pour quitter.\n")
if config.USE_CLOUD and not config.GROQ_API_KEY:
    print("KIRA : Avertissement -> GROQ_API_KEY manquante (cloud inactif).")
if not config.USE_CLOUD and not config.USE_LOCAL and not getattr(config, "USE_LITE", False):
    print("KIRA : Avertissement -> Aucun backend actif (cloud+local+lite désactivés).")
if config.USE_CLOUD and not config.GROQ_API_KEY and not config.USE_LOCAL and not getattr(config, "USE_LITE", False):
    print("KIRA : Avertissement -> Activez !local on / !lite on ou définissez GROQ_API_KEY.")

while True:
    user_input = input("Vous : ")
    raw_input = user_input.strip()
    command = raw_input.lower()
    if command in ["quit", "exit"]:
        print("KIRA : Arrêt demandé par le propriétaire.")
        break
    if command.startswith("!cloud on"):
        config.USE_CLOUD = True
        print("KIRA : Mode cloud Groq activé.")
        continue
    if command.startswith("!cloud off"):
        config.USE_CLOUD = False
        print("KIRA : Mode cloud désactivé – fonctionnement offline only.")
        continue
    if command.startswith("!local on"):
        config.USE_LOCAL = True
        print("KIRA : Mode local (Ollama) activé.")
        continue
    if command.startswith("!local off"):
        config.USE_LOCAL = False
        print("KIRA : Mode local (Ollama) désactivé.")
        continue
    if command == "!status":
        cloud_status = "Cloud activé" if config.USE_CLOUD else "Cloud désactivé"
        local_status = "Local activé" if config.USE_LOCAL else "Local désactivé"
        lite_status = "Lite activé" if getattr(config, "USE_LITE", False) else "Lite désactivé"
        perf_mode = getattr(config, "PERF_MODE", "balanced")
        print(
            "KIRA : Statut actuel -> "
            f"{cloud_status} | {local_status} | {lite_status} | Perf : {perf_mode} | Modèle Groq : {config.GROQ_MODEL}"
        )
        continue
    if command.startswith("!lite"):
        parts = raw_input.split()
        if len(parts) == 1:
            status = "activé" if getattr(config, "USE_LITE", False) else "désactivé"
            print(f"KIRA : Mode lite -> {status}")
        else:
            mode = parts[1].lower()
            if mode in ["on", "off"]:
                config.USE_LITE = mode == "on"
                log("lite_mode_changed", {"enabled": config.USE_LITE})
                status = "activé" if config.USE_LITE else "désactivé"
                print(f"KIRA : Mode lite -> {status}")
            else:
                print("KIRA : Usage : !lite on|off")
        continue
    if command.startswith("!micro"):
        parts = raw_input.split(maxsplit=2)
        if len(parts) == 1 or parts[1].lower() == "status":
            status = "activé" if getattr(config, "MICRO_LLM_ENABLED", False) else "désactivé"
            mode = getattr(config, "MICRO_LLM_MODE", "endpoint")
            endpoint = getattr(config, "MICRO_LLM_ENDPOINT", "")
            cmd = getattr(config, "MICRO_LLM_COMMAND", "")
            print(
                "KIRA : Micro-LLM -> "
                f"{status} | mode={mode} | endpoint={endpoint or 'N/A'} | cmd={'set' if cmd else 'N/A'}"
            )
        else:
            sub = parts[1].lower()
            if sub in ["on", "off"]:
                config.MICRO_LLM_ENABLED = sub == "on"
                log("micro_llm_enabled", {"enabled": config.MICRO_LLM_ENABLED})
                print(f"KIRA : Micro-LLM -> {'activé' if config.MICRO_LLM_ENABLED else 'désactivé'}")
            elif sub == "endpoint" and len(parts) > 2:
                config.MICRO_LLM_MODE = "endpoint"
                config.MICRO_LLM_ENDPOINT = parts[2].strip()
                log("micro_llm_endpoint_set", {"endpoint": config.MICRO_LLM_ENDPOINT})
                print(f"KIRA : Micro-LLM endpoint -> {config.MICRO_LLM_ENDPOINT}")
            elif sub == "command" and len(parts) > 2:
                config.MICRO_LLM_MODE = "command"
                config.MICRO_LLM_COMMAND = parts[2].strip()
                log("micro_llm_command_set", {"command": "set"})
                print("KIRA : Micro-LLM command -> défini")
            else:
                print("KIRA : Usage : !micro on|off|status|endpoint <url>|command <cmd>")
        continue
    if command.startswith("!perf"):
        parts = raw_input.split()
        if len(parts) == 1 or (len(parts) > 1 and parts[1].lower() in ["status", "show"]):
            settings = config.get_perf_settings()
            print(
                "KIRA : Mode perf -> "
                f"{config.PERF_MODE} | memoire_query={settings.get('memory_query')} "
                f"memoire_write={settings.get('memory_write')} "
                f"observation_query={settings.get('observation_query')} "
                f"auto_tasks={settings.get('auto_tasks')} "
                f"groq_max_tokens={settings.get('groq_max_tokens')} "
                f"local_num_predict={settings.get('local_num_predict')}"
            )
        else:
            mode = parts[1].lower()
            if mode in config.PERF_PRESETS:
                config.PERF_MODE = mode
                log("perf_mode_changed", {"mode": mode})
                settings = config.get_perf_settings()
                print(
                    "KIRA : Mode perf -> "
                    f"{mode} | memoire_query={settings.get('memory_query')} "
                    f"memoire_write={settings.get('memory_write')} "
                    f"observation_query={settings.get('observation_query')} "
                    f"auto_tasks={settings.get('auto_tasks')} "
                    f"groq_max_tokens={settings.get('groq_max_tokens')} "
                    f"local_num_predict={settings.get('local_num_predict')}"
                )
            else:
                print("KIRA : Usage : !perf fast|balanced|full|max")
        continue
    if command.startswith("!autonomy"):
        parts = raw_input.split()
        if len(parts) == 1:
            status = "activee" if getattr(config, "AUTONOMY_ENABLED", True) else "desactivee"
            count = getattr(config, "AUTONOMY_SUGGESTION_COUNT", 3)
            print(f"KIRA : Autonomie -> {status} | propositions={count}")
        else:
            mode = parts[1].lower()
            if mode in ["on", "off"]:
                config.AUTONOMY_ENABLED = mode == "on"
                log("autonomy_mode_changed", {"enabled": config.AUTONOMY_ENABLED})
                status = "activee" if config.AUTONOMY_ENABLED else "desactivee"
                print(f"KIRA : Autonomie -> {status}")
            elif mode.isdigit():
                config.AUTONOMY_SUGGESTION_COUNT = int(mode)
                log("autonomy_suggestions_changed", {"count": config.AUTONOMY_SUGGESTION_COUNT})
                print(f"KIRA : Autonomie -> propositions={config.AUTONOMY_SUGGESTION_COUNT}")
            else:
                print("KIRA : Usage : !autonomy on|off|<nombre>")
        continue
    if command.startswith("!style"):
        parts = raw_input.split()
        if len(parts) == 1:
            print(f"KIRA : Style -> {getattr(config, 'STYLE_MODE', 'natural')}")
        else:
            mode = parts[1].lower()
            if mode in ["natural", "strict"]:
                config.STYLE_MODE = mode
                log("style_mode_changed", {"mode": mode})
                print(f"KIRA : Style -> {mode}")
            else:
                print("KIRA : Usage : !style natural|strict")
        continue
    if command.startswith("!learning") or command.startswith("!learn"):
        parts = raw_input.split()
        if len(parts) == 1:
            status = "activee" if getattr(config, "LEARNING_ENABLED", True) else "desactivee"
            print(f"KIRA : Apprentissage -> {status}")
        else:
            mode = parts[1].lower()
            if mode in ["on", "off"]:
                config.LEARNING_ENABLED = mode == "on"
                log("learning_mode_changed", {"enabled": config.LEARNING_ENABLED})
                status = "activee" if config.LEARNING_ENABLED else "desactivee"
                print(f"KIRA : Apprentissage -> {status}")
            else:
                print("KIRA : Usage : !learning on|off")
        continue
    if command.startswith("!snapshot create"):
        desc = raw_input[len("!snapshot create") :].strip()
        snapshot_id = create_snapshot(description=desc or "Manuelle")
        print(f"KIRA : Snapshot créé -> {snapshot_id}")
        continue
    if command == "!snapshot list":
        snapshots = list_snapshots()
        if not snapshots:
            print("KIRA : Aucun snapshot disponible.")
        else:
            print("KIRA : Snapshots disponibles (plus récent en haut) :")
            for s in snapshots:
                print(f"  - {s['id']} | {s['timestamp']} | {s.get('description', 'Sans description')}")
        continue
    if command.startswith("!snapshot rollback "):
        snapshot_id = raw_input[len("!snapshot rollback ") :].strip()
        result = rollback_to(snapshot_id)
        print(f"KIRA : {result}")
        print("KIRA : Redémarrez le programme pour appliquer le rollback.")
        continue
    if command == "!observation veille":
        result = observe_all()
        print("KIRA : Veille terminée.")
        print(result)
        continue
    if command.startswith("!observation auto "):
        parts = command.split()
        if len(parts) > 2 and parts[2] == "on":
            config.OBSERVATION_AUTO_ENABLED = True
            print("KIRA : Mode veille automatique activé.")
        elif len(parts) > 2 and parts[2] == "off":
            config.OBSERVATION_AUTO_ENABLED = False
            print("KIRA : Mode veille automatique désactivé.")
        else:
            print("KIRA : Usage : !observation auto on|off")
        continue
    if command.startswith("!observation interval "):
        parts = raw_input.split()
        if len(parts) > 2:
            try:
                new_interval = int(parts[2])
                config.OBSERVATION_AUTO_INTERVAL_HOURS = new_interval
                print(
                    "KIRA : Intervalle de veille automatique réglé à "
                    f"{new_interval} heures."
                )
            except ValueError:
                print("KIRA : Usage : !observation interval 12")
        else:
            print("KIRA : Usage : !observation interval 12")
        continue
    if command.startswith("!observation autovalidate"):
        parts = raw_input.split()
        if len(parts) > 2 and parts[2].lower() in ["on", "off"]:
            config.OBSERVATION_AUTO_VALIDATE = parts[2].lower() == "on"
            print(
                f"KIRA : Auto-validation observation -> "
                f"{'activée' if config.OBSERVATION_AUTO_VALIDATE else 'désactivée'}"
            )
        elif len(parts) > 2:
            try:
                score = float(parts[2])
                config.OBSERVATION_AUTO_VALIDATE = True
                config.OBSERVATION_AUTO_VALIDATE_MIN_SCORE = score
                print(
                    "KIRA : Auto-validation observation -> activée | "
                    f"seuil = {score:.2f}"
                )
            except ValueError:
                print("KIRA : Usage : !observation autovalidate on|off|0.8")
        else:
            status = "activée" if config.OBSERVATION_AUTO_VALIDATE else "désactivée"
            print(
                "KIRA : Auto-validation observation -> "
                f"{status} | seuil = {config.OBSERVATION_AUTO_VALIDATE_MIN_SCORE:.2f}"
            )
        continue
    if command == "!observation status":
        status = "activé" if config.OBSERVATION_AUTO_ENABLED else "désactivé"
        last_time = get_last_veille_time()
        last_display = (
            last_time.isoformat()
            if last_time != datetime.datetime.min
            else "Jamais"
        )
        print(
            "KIRA : Veille auto : "
            f"{status} | Intervalle : {config.OBSERVATION_AUTO_INTERVAL_HOURS} heures "
            f"| Dernière veille : {last_display}"
        )
        continue
    if command.startswith("!observation add "):
        url = raw_input[len("!observation add ") :].strip()
        load_sources()
        if not url:
            print("KIRA : Usage : !observation add <url>")
            continue
        with open(OBSERVATION_SOURCES_FILE, "a", encoding="utf-8") as f:
            f.write(url + "\n")
        print(f"KIRA : Source ajoutée : {url}")
        continue
    if command == "!observation sources":
        sources = load_sources()
        if not sources:
            print("KIRA : Sources configurées : Aucune")
        else:
            lines = [
                f"{s['url']} | fiabilité {s['score']:.2f}" for s in sources if s.get("url")
            ]
            print("KIRA : Sources configurées :", "\n".join(lines))
        continue
    if command == "!observation validate":
        result = validate_pending()
        print(f"KIRA : {result}")
        continue
    if command == "!observation reject":
        result = reject_last()
        print(f"KIRA : {result}")
        continue
    if command.startswith("!evolution auto "):
        parts = command.split()
        if len(parts) > 2 and parts[2] == "on":
            config.EVOLUTION_AUTO_ENABLED = True
            print("KIRA : Mode évolution automatique activé.")
        elif len(parts) > 2 and parts[2] == "off":
            config.EVOLUTION_AUTO_ENABLED = False
            print("KIRA : Mode évolution automatique désactivé.")
        else:
            print("KIRA : Usage : !evolution auto on|off")
        continue
    if command.startswith("!evolution threshold "):
        parts = raw_input.split()
        if len(parts) > 2:
            try:
                new_thresh = int(parts[2])
                config.EVOLUTION_AUTO_THRESHOLD = new_thresh
                print(
                    f"KIRA : Seuil d'évolution automatique réglé à {new_thresh} interactions."
                )
            except ValueError:
                print("KIRA : Usage : !evolution threshold 20")
        else:
            print("KIRA : Usage : !evolution threshold 20")
        continue
    if command.startswith("!evolution analyze"):
        limit = 20
        parts = raw_input.split()
        if len(parts) > 2:
            try:
                limit = int(parts[2])
            except ValueError:
                pass
        print("KIRA : Analyse méta en cours...")
        analysis = analyze_recent(limit=limit)
        print("\n=== ANALYSE D'ÉVOLUTION ===\n")
        print(analysis)
        continue
    if command.startswith("!evolution propose"):
        desc = raw_input[len("!evolution propose") :].strip()
        if not desc:
            print(
                "KIRA : Veuillez fournir une description "
                "(ex: !evolution propose Amélioration orchestration)"
            )
            continue
        # On suppose que vous avez d'abord fait une analyse
        # Ici, on peut réutiliser une analyse récente ou en générer une légère
        recent_analysis = analyze_recent(limit=10)
        proposal_id, patch = propose_patch_from_analysis(
            recent_analysis, trigger="manual", description=desc
        )
        print(f"KIRA : Proposition d'évolution créée -> {proposal_id}")
        print("\nContenu de la proposition :\n")
        print(patch)
        print(f"\nKIRA : Fichiers dans evolutions/{proposal_id}/")
        print("KIRA : Avant toute application : créez un snapshot (!snapshot create).")
        continue
    if command == "!evolution list":
        if not os.path.exists(EVOLUTION_DIR):
            print("KIRA : Aucune proposition d'évolution.")
            continue
        proposals = [
            d
            for d in os.listdir(EVOLUTION_DIR)
            if os.path.isdir(os.path.join(EVOLUTION_DIR, d))
        ]
        proposals.sort(reverse=True)
        print("KIRA : Propositions d'évolution disponibles :")
        for p in proposals:
            manifest_path = os.path.join(EVOLUTION_DIR, p, "manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    print(
                        f"  - {p} | {m.get('timestamp', '')} | "
                        f"{m.get('description', 'Sans description')}"
                    )
        continue
    try:
        response = process_input(user_input)
    except Exception as e:
        log("runtime_error", {"error": str(e)})
        print("KIRA : Erreur interne. Consultez audit.log pour le detail.")
        continue
    print("\nKIRA :", response, "\n")
