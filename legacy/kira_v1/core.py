
import time

import config

from orchestrator import query_models
from memory import add_dialogue, prune_memory, search_memory, search_observation
from synthesis import build_consensus_prompt, select_best_response
from audit import log
from config import PROPRIETAIRE, PRINCIPES


def _get_perf_settings():
    try:
        return config.get_perf_settings()
    except Exception:
        return {}


def _truncate_text(text: str, max_chars: int):
    if not text:
        return ""
    if isinstance(max_chars, int) and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def _extract_summary_from_doc(doc: str):
    if not doc:
        return ""
    marker = "Summary:"
    if marker in doc:
        snippet = doc.split(marker, 1)[1].strip()
        return snippet.splitlines()[0].strip()
    return doc.strip().splitlines()[0][:400]


def _format_memory_context(results, max_items: int, max_chars: int):
    if not results:
        return ""
    documents = results.get("documents", [[]])[0] if results else []
    metadatas = results.get("metadatas", [[]])[0] if results else []
    lines = []
    for doc, meta in zip(documents, metadatas):
        if max_items and len(lines) >= max_items:
            break
        meta = meta or {}
        question = meta.get("question", "") if isinstance(meta, dict) else ""
        summary = meta.get("summary", "") if isinstance(meta, dict) else ""
        text = ""
        if summary:
            text = f"Q: {question}\nR: {summary}" if question else summary
        elif doc:
            text = _extract_summary_from_doc(str(doc))
        if text:
            lines.append(text.strip())
    joined = "\n\n".join(lines)
    return _truncate_text(joined, max_chars)


def _should_propose_actions(user_input: str):
    lowered = (user_input or "").lower()
    triggers = [
        "que me proposes",
        "que proposes",
        "quoi faire",
        "propose",
        "suggestion",
        "idees",
        "idées",
        "autonome",
        "autonomie",
    ]
    return any(t in lowered for t in triggers)


def _detect_owner_authorization(user_input: str):
    lowered = (user_input or "").lower()
    keywords = [
        "je valide",
        "je t'autorise",
        "je te donne l'autorisation",
        "c'est moi ton proprietaire",
        "cest moi ton proprietaire",
        "accordons",
        "autonomie maximale",
        "plus d'autonomie",
        "commencons",
        "commençons",
        "vas-y",
    ]
    return any(k in lowered for k in keywords)


def process_input(user_input: str):
    log("user_input", {"input": user_input})
    total_start = time.monotonic()
    timings = {}

    # Mémoire courte : contexte récent (simplifié)
    perf = _get_perf_settings()
    memory_query = perf.get("memory_query", True)
    memory_write = perf.get("memory_write", True)
    memory_items = perf.get("memory_items", 5)
    memory_max_chars = perf.get("memory_max_chars", 2000)
    observation_query = perf.get("observation_query", True)
    observation_items = perf.get("observation_items", 5)
    observation_max_chars = perf.get("observation_max_chars", 2000)
    auto_tasks = perf.get("auto_tasks", True)
    synthesis_mode = perf.get("synthesis_mode", config.SYNTHESIS_MODE)
    autonomy_enabled = getattr(config, "AUTONOMY_ENABLED", True)
    autonomy_suggestions = getattr(config, "AUTONOMY_SUGGESTION_COUNT", 3)
    style_mode = getattr(config, "STYLE_MODE", "natural")
    learning_enabled = getattr(config, "LEARNING_ENABLED", True)
    learning_force_recall = getattr(config, "LEARNING_FORCE_RECALL", True)
    learning_min_items = getattr(config, "LEARNING_MIN_ITEMS", 2)
    lite_mode = getattr(config, "USE_LITE", False) and not (config.USE_CLOUD or config.USE_LOCAL)
    lite_disable_auto = getattr(config, "LITE_DISABLE_AUTO_TASKS", False)
    lite_disable_evolution = getattr(config, "LITE_DISABLE_EVOLUTION_AUTO", False) or lite_disable_auto
    lite_disable_observation = getattr(config, "LITE_DISABLE_OBSERVATION_AUTO", False) or lite_disable_auto
    lite_disable_vector_memory = getattr(config, "LITE_DISABLE_VECTOR_MEMORY", False)
    lite_disable_obs_context = getattr(config, "LITE_DISABLE_OBSERVATION_CONTEXT", False)

    if learning_enabled:
        if not memory_write:
            memory_write = True
        if learning_force_recall and not memory_query:
            memory_query = True
        if memory_items < learning_min_items:
            memory_items = learning_min_items
    auto_evolution = auto_tasks and not (lite_mode and lite_disable_evolution)
    auto_observation = auto_tasks and not (lite_mode and lite_disable_observation)
    if lite_mode and lite_disable_vector_memory:
        memory_query = False
        memory_write = False
    if lite_mode and lite_disable_obs_context:
        observation_query = False

    context_text = ""
    if memory_query and memory_items > 0:
        t0 = time.monotonic()
        context = search_memory(user_input, n=memory_items)
        context_text = _format_memory_context(
            context, max_items=memory_items, max_chars=memory_max_chars
        )
        timings["memory_ms"] = int((time.monotonic() - t0) * 1000)

    if style_mode == "strict":
        style_hint = (
            "Réponds de manière claire, structurée et fidèle au Cahier des Charges. "
            "Garde un ton professionnel et concis."
        )
    else:
        style_hint = (
            "Adopte un ton naturel et conversationnel. Sois fluide, direct, humain, "
            "évite les formules rigides. Pose une seule question de clarification "
            "si nécessaire."
        )

    # Prompt système renforçant l’identité KIRA
    system_prompt = f"""
Tu es KIRA, système d’intelligence personnelle souveraine de {PROPRIETAIRE}.
Tu respectes strictement ces principes : {', '.join(PRINCIPES)}.
Tu ne fais jamais d’action silencieuse. Tu proposes, tu n’exécutes pas sans validation.
Tu réponds directement aux questions d’information sans demander de validation.
La validation est requise uniquement pour des actions concrètes (modification de fichiers,
exécution de commandes, appels réseau, achats, changements système).
Tu ne répètes pas la liste des principes à chaque réponse sauf si on te le demande.
Si le propriétaire accorde explicitement une autorisation d’autonomie, tu l’acceptes et
tu proposes un plan d’action concret et priorisé (sans te déclarer bloqué).
{style_hint}
    """

    full_prompt = system_prompt
    if context_text:
        full_prompt += "\nContexte memoire (resume) :\n" + context_text
    full_prompt += "\nQuestion : " + user_input

    if autonomy_enabled and _should_propose_actions(user_input):
        full_prompt += (
            "\n\nInstruction : la demande est ouverte. "
            f"Propose {autonomy_suggestions} actions concrètes, classées par priorité, "
            "avec bénéfices et risques. Demande validation uniquement pour les actions."
        )

    if autonomy_enabled and _detect_owner_authorization(user_input):
        log("owner_authorization_detected", {"input": user_input})
        full_prompt += (
            "\n\nInstruction : le propriétaire confirme l'autorisation d'autonomie. "
            "Propose un plan en 3 à 5 actions concrètes pour augmenter l'autonomie "
            "de KIRA, avec un ordre de priorité, et demande une validation unique "
            "pour lancer l'étape 1."
        )

    obs_context = ""
    if observation_query and observation_items > 0:
        t0 = time.monotonic()
        obs_context = search_observation(user_input, n=observation_items)
        obs_context = _truncate_text(obs_context, observation_max_chars)
        timings["observation_ms"] = int((time.monotonic() - t0) * 1000)
    if obs_context:
        full_prompt += "\n\nContexte observation (veille validee) :\n" + obs_context

    timings["prompt_chars"] = len(full_prompt)
    timings["memory_context_chars"] = len(context_text)
    timings["observation_context_chars"] = len(obs_context)

    t0 = time.monotonic()
    responses = query_models(full_prompt)
    timings["models_ms"] = int((time.monotonic() - t0) * 1000)

    if isinstance(responses, str):
        synthesis = responses
    else:
        candidates = {
            k: v for k, v in responses.items() if v and not str(v).startswith("[Erreur")
        }
        synthesis = ""
        if synthesis_mode == "consensus" and len(candidates) >= 2:
            consensus_prompt = build_consensus_prompt(user_input, candidates)
            t0 = time.monotonic()
            consensus = query_models(consensus_prompt)
            timings["consensus_ms"] = int((time.monotonic() - t0) * 1000)
            if isinstance(consensus, str):
                synthesis = consensus
            elif isinstance(consensus, dict):
                synthesis = select_best_response(consensus)
            if synthesis:
                log("synthesis_consensus", {"candidates": len(candidates)})
        if not synthesis:
            if candidates:
                synthesis = select_best_response(candidates)
            else:
                synthesis = select_best_response(responses)

    # Sauvegarde en mémoire longue
    if memory_write:
        t0 = time.monotonic()
        add_dialogue(user_input, synthesis)
        if config.MEMORY_AUTO_PRUNE and config.MEMORY_MAX_ITEMS > 0:
            prune_memory(config.MEMORY_MAX_ITEMS)
        timings["memory_write_ms"] = int((time.monotonic() - t0) * 1000)

    if auto_evolution:
        t0 = time.monotonic()
        try:
            from evolution import (
                analysis_has_opportunity,
                analyze_recent,
                increment_interaction,
                propose_patch_from_analysis,
            )
            count = increment_interaction()
            if (
                config.EVOLUTION_AUTO_ENABLED
                and config.EVOLUTION_AUTO_THRESHOLD > 0
                and count % config.EVOLUTION_AUTO_THRESHOLD == 0
            ):
                analysis = analyze_recent()
                synthesis += (
                    f"\n\n=== ANALYSE D'EVOLUTION AUTOMATIQUE "
                    f"(tous les {config.EVOLUTION_AUTO_THRESHOLD} dialogues) ===\n"
                )
                synthesis += "Analyse sauvegardee dans evolutions/auto/.\n"
                if analysis_has_opportunity(analysis):
                    proposal_id, patch = propose_patch_from_analysis(
                        analysis, trigger="auto"
                    )
                    synthesis += f"Proposition #{proposal_id} generee :\n{patch}\n"
                    synthesis += (
                        "KIRA : Je vous soumets cette evolution pour examen. "
                        "Creez un snapshot avant tout test."
                    )
                else:
                    synthesis += (
                        "KIRA : Aucune opportunite claire detectee pour une proposition."
                    )
        except Exception as e:
            log("evolution_auto_error", {"error": str(e)})
        timings["auto_evolution_ms"] = int((time.monotonic() - t0) * 1000)

    if auto_observation:
        t0 = time.monotonic()
        try:
            from observation import auto_validate_by_score, observe_all, should_trigger_auto_veille
            if should_trigger_auto_veille():
                veille_results = observe_all(propose_validation=True)
                synthesis += "\n\n=== VEILLE AUTOMATIQUE DECLENCHEE (intervalle ecoule) ===\n"
                synthesis += f"{veille_results}\n"
                synthesis += (
                    "KIRA : Cette veille a ete effectuee automatiquement. "
                    "Validez les integrations si souhaitees."
                )
                if config.OBSERVATION_AUTO_VALIDATE:
                    validated = auto_validate_by_score(
                        config.OBSERVATION_AUTO_VALIDATE_MIN_SCORE
                    )
                    if validated:
                        synthesis += (
                            f"\nKIRA : Auto-validation de {validated} observations "
                            f"(score >= {config.OBSERVATION_AUTO_VALIDATE_MIN_SCORE:.2f})."
                        )
                log(
                    "observation_auto_triggered",
                    {"interval_hours": config.OBSERVATION_AUTO_INTERVAL_HOURS},
                )
        except Exception as e:
            log("observation_auto_error", {"error": str(e)})
        timings["auto_observation_ms"] = int((time.monotonic() - t0) * 1000)

    timings["response_chars"] = len(synthesis or "")
    timings["perf_mode"] = config.PERF_MODE
    timings["synthesis_mode"] = synthesis_mode
    timings["total_ms"] = int((time.monotonic() - total_start) * 1000)
    log("perf_timing", timings)

    return synthesis
