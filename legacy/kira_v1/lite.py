import math
import re

import config
from lite_store import (
    add_history,
    add_note,
    add_todo,
    complete_todo,
    get_preferences,
    get_recent_history,
    list_notes,
    list_todos,
    load_store,
    save_store,
    set_preference,
)
from micro_llm import generate as micro_generate


def _extract_block(text: str, start_label: str, end_labels: list[str]):
    start = text.find(start_label)
    if start == -1:
        return ""
    start += len(start_label)
    end = len(text)
    for label in end_labels:
        idx = text.find(label, start)
        if idx != -1:
            end = min(end, idx)
    return text[start:end].strip()


def _extract_question(prompt: str):
    marker = "Question :"
    if marker in prompt:
        return prompt.split(marker)[-1].strip()
    return prompt.strip().splitlines()[-1] if prompt else ""


def _extract_count(prompt: str, default: int):
    match = re.search(r"Propose\s+(\d+)\s+actions", prompt)
    if match:
        try:
            return max(1, int(match.group(1)))
        except ValueError:
            return default
    return default


def _is_summary_prompt(prompt: str):
    lowered = (prompt or "").lower()
    return "résume ce contenu" in lowered or "resume ce contenu" in lowered or "résumé observé" in lowered


def _summarize_content(content: str):
    cleaned = " ".join((content or "").split())
    if not cleaned:
        return "Résumé observé :\nObservation interprétée : Aucune."
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    max_sentences = getattr(config, "LITE_SUMMARY_SENTENCES", 3)
    summary = " ".join(sentences[: max(1, max_sentences)]).strip()
    if not summary:
        summary = cleaned[:400]
    signal_keywords = [
        "hausse",
        "baisse",
        "augmentation",
        "diminution",
        "tendance",
        "lancement",
        "annonce",
        "partenariat",
        "croissance",
        "ralentissement",
        "accélération",
        "acceleration",
    ]
    interpretation = "Observation interprétée : Aucune."
    if any(k in cleaned.lower() for k in signal_keywords):
        interpretation = "Observation interprétée : Signal possible (tendance/annonce)."
    return f"{summary}\n{interpretation}"


def _summarize_from_prompt(prompt: str):
    marker = "Contenu :"
    if marker in prompt:
        content = prompt.split(marker, 1)[1]
    else:
        content = prompt
    return _summarize_content(content)


def _is_greeting(text: str):
    lowered = (text or "").lower()
    greetings = ["bonjour", "salut", "hello", "bjr", "yo", "coucou", "bonsoir", "bsr"]
    return any(g in lowered for g in greetings)


def _is_capability_question(text: str):
    lowered = (text or "").lower()
    patterns = [
        "que peux tu faire",
        "que peux-tu faire",
        "tu peux faire quoi",
        "que sais tu faire",
        "que sais-tu faire",
        "capable de quoi",
        "tes capacites",
        "tes capacités",
        "que proposes tu",
        "que proposes-tu",
    ]
    return any(p in lowered for p in patterns)


def _capabilities_reply():
    return (
        "Je suis en mode léger (sans Groq/Ollama). Voici ce que je peux faire vite :\n"
        "1) Répondre à des questions simples et donner des explications courtes.\n"
        "2) Proposer des actions concrètes à valider (ex: réglages, options, étapes).\n"
        "3) Résumer ou reformuler un texte que tu me donnes.\n"
        "4) Gestion perso : todo / notes / préférences.\n"
        "5) Maths, logique et physique de base (formules simples).\n"
        "6) Aide code légère (templates, checklists).\n"
        "7) Piloter KIRA via commandes :\n"
        "   - !perf fast|balanced|full|max\n"
        "   - !style natural|strict\n"
        "   - !learning on|off\n"
        "   - !autonomy on|off|<nombre>\n"
        "   - !lite on|off\n"
        "   - !snapshot create|list|rollback\n"
        "   - !observation veille|validate|reject\n"
        "8) Discuter et clarifier tes besoins.\n"
        "Limite : réponses plus simples, pas d’IA lourde.\n"
        "Dis‑moi ce que tu veux faire maintenant."
    )


def _parse_assignment_vars(text: str):
    pairs = re.findall(r"([a-zA-Z]+)\s*=\s*([0-9]+(?:[\\.,][0-9]+)?)", text)
    values = {}
    for key, raw in pairs:
        try:
            values[key.lower()] = float(raw.replace(",", "."))
        except ValueError:
            continue
    return values


def _safe_eval(expr: str):
    import ast

    allowed_funcs = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)):
                return node.value
            raise ValueError("const")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            if isinstance(node.op, ast.Not):
                return not bool(_eval(node.operand))
            return +_eval(node.operand) if isinstance(node.op, ast.UAdd) else -_eval(node.operand)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [_eval(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(bool(v) for v in values)
            return any(bool(v) for v in values)
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
        ):
            left = _eval(node.left)
            right = _eval(node.right)
            return {
                ast.Add: left + right,
                ast.Sub: left - right,
                ast.Mult: left * right,
                ast.Div: left / right,
                ast.Pow: left**right,
                ast.Mod: left % right,
            }[type(node.op)]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = allowed_funcs.get(node.func.id)
            if not fn:
                raise ValueError("func")
            args = [_eval(arg) for arg in node.args]
            return fn(*args)
        if isinstance(node, ast.Name):
            if node.id in allowed_funcs and isinstance(allowed_funcs[node.id], (int, float)):
                return allowed_funcs[node.id]
        raise ValueError("node")

    expr = expr.replace("^", "**")
    tree = ast.parse(expr, mode="eval")
    return _eval(tree)


def _try_math(user_input: str):
    text = (user_input or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("calcule"):
        expr = text[len("calcule") :].strip()
    elif any(ch.isdigit() for ch in text) and any(op in text for op in "+-*/^"):
        expr = text
    else:
        return None
    try:
        value = _safe_eval(expr)
        return f"Résultat : {value}"
    except Exception:
        return None


def _try_logic(user_input: str):
    text = (user_input or "").strip().lower()
    if not any(tok in text for tok in [" et ", " ou ", " non ", "and", "or", "not"]):
        return None
    expr = (
        text.replace("vrai", "True")
        .replace("faux", "False")
        .replace(" et ", " and ")
        .replace(" ou ", " or ")
        .replace(" non ", " not ")
    )
    try:
        value = _safe_eval(expr)
        if isinstance(value, (bool, int, float)):
            return f"Évaluation logique : {bool(value)}"
    except Exception:
        return None
    return None


def _try_physics(user_input: str):
    text = (user_input or "").lower()
    vars_map = _parse_assignment_vars(text)
    if not vars_map:
        return None
    # F = m * a
    if ("f=ma" in text or "force" in text) and "m" in vars_map and "a" in vars_map:
        f_val = vars_map["m"] * vars_map["a"]
        return f"Force F = {f_val} N"
    # v = d / t
    if ("v=" in text or "vitesse" in text) and "d" in vars_map and "t" in vars_map:
        v_val = vars_map["d"] / vars_map["t"]
        return f"Vitesse v = {v_val} m/s"
    # E = m * c^2
    if ("e=mc2" in text or "energie" in text or "énergie" in text) and "m" in vars_map:
        c = 299_792_458
        e_val = vars_map["m"] * (c**2)
        return f"Énergie E = {e_val} J"
    # p = m * v
    if ("p=mv" in text or "quantité de mouvement" in text) and "m" in vars_map and "v" in vars_map:
        p_val = vars_map["m"] * vars_map["v"]
        return f"Quantité de mouvement p = {p_val} kg·m/s"
    return None


def _try_code(user_input: str):
    text = (user_input or "").lower()
    if "code" not in text and "python" not in text and "javascript" not in text and "c#" not in text:
        return None
    if "python" in text:
        return "Exemple Python (fonction) :\n```\n def ma_fonction(x):\n     return x * 2\n```"
    if "javascript" in text or "js" in text:
        return "Exemple JS (fonction) :\n```\n function maFonction(x) {\n   return x * 2;\n }\n```"
    if "c#" in text or "csharp" in text:
        return "Exemple C# (méthode) :\n```\n int Double(int x) {\n     return x * 2;\n }\n```"
    return "Dis-moi le langage et l’objectif, je te propose un squelette."


def _try_personal(store: dict, user_input: str):
    text = (user_input or "").strip()
    lowered = text.lower()
    # Add todo
    match = re.search(
        r"(ajoute|ajouter|ajoutes)\s+(?:une\s+)?(?:tache|tâche|todo)\s*[:\-]?\s*(.+)",
        lowered,
    )
    if match:
        add_todo(store, text[match.start(2) :].strip())
        save_store(store)
        return "Tâche ajoutée."
    match = re.search(
        r"(ajoute|ajouter|ajoutes)\s+(.+?)\s+(?:dans\s+)?(?:la\s+)?liste\s+des\s+(taches|tâches|todos)",
        lowered,
    )
    if match:
        add_todo(store, text[match.start(2) : match.end(2)].strip())
        save_store(store)
        return "Tâche ajoutée."
    # List todos
    if re.search(
        r"(liste|affiche|montre)\s+(?:mes\s+|des\s+|de\s+)?(taches|tâches|todos)",
        lowered,
    ):
        todos = list_todos(store)
        if not todos:
            return "Aucune tâche en cours."
        lines = ["Tâches en cours :"]
        for idx, todo in enumerate(todos, start=1):
            lines.append(f"{idx}) {todo.get('text','')}")
        return "\n".join(lines)
    # Remove todo by index
    match = re.search(r"(retire|supprime|enleve|enlève)\s+(?:la\s+)?tache\s*(\d+)", lowered)
    if match:
        index = int(match.group(2))
        if complete_todo(store, index):
            save_store(store)
            return "Tâche supprimée."
        return "Numéro de tâche invalide."
    match = re.search(r"(retire|supprime|enleve|enlève)\s+le\s+(\d+)", lowered)
    if match:
        index = int(match.group(2))
        if complete_todo(store, index):
            save_store(store)
            return "Tâche supprimée."
        return "Numéro de tâche invalide."
    # Remove todo by text
    match = re.search(r"(retire|supprime|enleve|enlève)\s+(.+)", lowered)
    if match:
        target = match.group(2).strip()
        todos = list_todos(store, include_done=True)
        for idx, todo in enumerate(todos, start=1):
            if target in todo.get("text", "").lower():
                if complete_todo(store, idx):
                    save_store(store)
                    return f"Tâche supprimée : {todo.get('text','')}"
        return "Je n’ai pas trouvé cette tâche."
    # Complete todo
    match = re.search(r"(termine|complete|finis)\s+(?:tache|tâche|todo)\s*(\d+)", lowered)
    if match:
        if complete_todo(store, int(match.group(2))):
            save_store(store)
            return "Tâche marquée comme terminée."
        return "Numéro de tâche invalide."
    # Add note
    match = re.search(r"(ajoute|ajouter)\s+(?:une\s+)?note\s*[:\-]?\s*(.+)", lowered)
    if match:
        add_note(store, text[match.start(2) :].strip())
        save_store(store)
        return "Note ajoutée."
    # List notes
    if re.search(r"(liste|affiche|montre)\s+(?:mes\s+)?notes", lowered):
        notes = list_notes(store)
        if not notes:
            return "Aucune note."
        lines = ["Dernières notes :"]
        for idx, note in enumerate(notes, start=1):
            lines.append(f"{idx}) {note.get('text','')}")
        return "\n".join(lines)
    # Preferences
    match = re.search(r"(preference|préférence)\\s*[:\\-]?\\s*([^=]+)=(.+)", lowered)
    if match:
        key = match.group(2).strip()
        val = match.group(3).strip()
        set_preference(store, key, val)
        save_store(store)
        return f"Préférence enregistrée : {key} = {val}"
    if "mes preferences" in lowered or "mes préférences" in lowered:
        prefs = get_preferences(store)
        if not prefs:
            return "Aucune préférence enregistrée."
        lines = ["Préférences :"]
        for key, val in prefs.items():
            lines.append(f"- {key} = {val}")
        return "\n".join(lines)
    return None


def _try_feedback(store: dict, user_input: str):
    text = (user_input or "").strip().lower()
    if not text:
        return None
    stop_patterns = [
        "arrête de dire",
        "arrete de dire",
        "stop de dire",
        "ne dis plus",
        "ne dit plus",
        "tout le temps",
        "toujours",
        "répète",
        "repete",
    ]
    if any(p in text for p in stop_patterns):
        set_preference(store, "suppress_mode_notice", "true")
        save_store(store)
        return "Compris. Je ne le répète plus."
    return None


def _keyword_actions(user_input: str):
    lowered = (user_input or "").lower()
    if any(k in lowered for k in ["lent", "latence", "rapide", "vitesse", "lag"]):
        return [
            (
                "Passer en mode perf rapide (`!perf fast`)",
                "Réduit fortement la latence sans couper l’apprentissage.",
                "Réponses plus courtes et contexte réduit.",
            ),
            (
                "Désactiver l’observation contextuelle (`!perf balanced` + observation_query=False)",
                "Moins d’IO et d’embeddings par requête.",
                "Perte des signaux issus de la veille validée.",
            ),
            (
                "Limiter la taille de sortie (tokens max à 600–800)",
                "Réponses plus rapides et plus stables.",
                "Moins de détails dans les réponses longues.",
            ),
        ]
    if any(k in lowered for k in ["autonomie", "autonome", "autonome"]):
        return [
            (
                "Activer l’autonomie (`!autonomy on`)",
                "KIRA propose des actions sans attente explicite.",
                "Peut proposer des actions plus souvent.",
            ),
            (
                "Augmenter les propositions (`!autonomy 5`)",
                "Plus d’options à valider pour accélérer la décision.",
                "Risque d’info‑overload.",
            ),
            (
                "Activer les tâches auto en mode balanced",
                "Permet analyse/veille en arrière‑plan.",
                "Peut consommer plus de temps.",
            ),
        ]
    if any(k in lowered for k in ["apprend", "apprentissage", "memoire", "mémoire"]):
        return [
            (
                "Forcer l’apprentissage (`!learning on`)",
                "Mémorise chaque interaction.",
                "Peut accumuler des infos inutiles.",
            ),
            (
                "Limiter le contexte mémoire",
                "Rappel plus rapide et stable.",
                "Moins de détails historiques.",
            ),
            (
                "Créer un fichier de préférences",
                "Apprentissage explicite et durable.",
                "Nécessite une définition initiale.",
            ),
        ]
    return [
        (
            "Clarifier l’objectif principal",
            "Permet une réponse ciblée et efficace.",
            "Nécessite une précision de ta part.",
        ),
        (
            "Lister 2–3 options concrètes",
            "Accélère la prise de décision.",
            "Peut rester générique sans contexte.",
        ),
        (
            "Exécuter une première étape minimale",
            "Démarre rapidement.",
            "Peut nécessiter ajustements ensuite.",
        ),
    ]


def _format_actions(actions: list[tuple[str, str, str]], count: int):
    lines = ["Voici mes propositions prioritaires :"]
    for idx, (action, benefit, risk) in enumerate(actions[:count], start=1):
        lines.append(f"{idx}) {action}")
        lines.append(f"   Bénéfice : {benefit}")
        lines.append(f"   Risque : {risk}")
    lines.append("Dis‑moi laquelle tu valides (ex: 1, 2 ou 3).")
    return "\n".join(lines)


def _default_reply(user_input: str, memory_context: str, obs_context: str):
    if _is_capability_question(user_input):
        return _capabilities_reply()
    lines = []
    is_greeting = _is_greeting(user_input)
    if is_greeting:
        lines.append("Bonsoir !")
    suppress_mode_notice = False
    try:
        prefs = get_preferences(load_store())
        suppress_mode_notice = bool(prefs.get("suppress_mode_notice"))
    except Exception:
        suppress_mode_notice = False
    if not suppress_mode_notice:
        lines.append("Je tourne en mode léger (sans Groq/Ollama).")
    if memory_context:
        snippet = memory_context[:200].replace("\n", " ").strip()
        if snippet and "erreur" not in snippet.lower() and "aucune réponse disponible" not in snippet.lower():
            lines.append(f"Je me souviens : {snippet}")
    if obs_context:
        snippet = obs_context[:200].replace("\n", " ").strip()
        if snippet:
            lines.append(f"Veille utile : {snippet}")
    if user_input and not is_greeting:
        lines.append("Qu’est‑ce que tu veux faire exactement ?")
        lines.append(f"Contexte : “{user_input}”")
    else:
        lines.append("Tu veux qu’on fasse quoi ?")
    return "\n".join(lines)


def _build_micro_prompt(user_input: str, store: dict, memory_context: str, obs_context: str):
    prefs = get_preferences(store)
    history = get_recent_history(store, limit=4)
    history_text = "\n".join([f"Q: {h.get('q','')}\nR: {h.get('r','')}" for h in history])
    pref_text = ", ".join([f"{k}={v}" for k, v in prefs.items()]) if prefs else "Aucune"
    prompt = (
        "Tu es KIRA, assistant personnel conversationnel. "
        "Réponds de manière naturelle, claire et concise.\n"
        f"Préférences: {pref_text}\n"
    )
    if history_text:
        prompt += f"Historique récent:\n{history_text}\n"
    if memory_context:
        prompt += f"Contexte mémoire:\n{memory_context}\n"
    if obs_context:
        prompt += f"Contexte veille:\n{obs_context}\n"
    prompt += f"Question: {user_input}\nRéponse:"
    return prompt


def _autonomy_plan():
    return (
        "Plan d’autonomie (version lite) :\n"
        "1) Activer `!autonomy on` et fixer le niveau de propositions.\n"
        "2) Définir un périmètre d’actions autorisées (fichiers, commandes, réseau).\n"
        "3) Mettre un journal clair des actions et décisions.\n"
        "4) Ajouter une routine d’amélioration hebdomadaire validée.\n"
        "Valides‑tu l’étape 1 ?"
    )


def respond(prompt: str):
    if _is_summary_prompt(prompt):
        summary = _summarize_from_prompt(prompt)
        return summary
    user_input = _extract_question(prompt)
    memory_context = _extract_block(
        prompt,
        "Contexte memoire (resume) :",
        ["Contexte observation (veille validee) :", "Question :"],
    )
    obs_context = _extract_block(
        prompt,
        "Contexte observation (veille validee) :",
        ["Question :"],
    )

    store = load_store()
    feedback = _try_feedback(store, user_input)
    if feedback:
        add_history(store, user_input, feedback)
        save_store(store)
        return feedback

    personal = _try_personal(store, user_input)
    if personal:
        add_history(store, user_input, personal)
        save_store(store)
        return personal

    math_reply = _try_math(user_input)
    if math_reply:
        add_history(store, user_input, math_reply)
        save_store(store)
        return math_reply

    physics_reply = _try_physics(user_input)
    if physics_reply:
        add_history(store, user_input, physics_reply)
        save_store(store)
        return physics_reply

    logic_reply = _try_logic(user_input)
    if logic_reply:
        add_history(store, user_input, logic_reply)
        save_store(store)
        return logic_reply

    code_reply = _try_code(user_input)
    if code_reply:
        add_history(store, user_input, code_reply)
        save_store(store)
        return code_reply

    micro_reply = ""
    if getattr(config, "MICRO_LLM_ENABLED", False):
        micro_prompt = _build_micro_prompt(user_input, store, memory_context, obs_context)
        micro_reply = micro_generate(micro_prompt)
    if micro_reply:
        add_history(store, user_input, micro_reply)
        save_store(store)
        return micro_reply

    if "Instruction : le propriétaire confirme l'autorisation d'autonomie" in prompt:
        response = _autonomy_plan()
    elif "Instruction : la demande est ouverte." in prompt:
        count = _extract_count(prompt, getattr(config, "AUTONOMY_SUGGESTION_COUNT", 3))
        actions = _keyword_actions(user_input)
        response = _format_actions(actions, count)
    else:
        response = _default_reply(user_input, memory_context, obs_context)

    max_chars = getattr(config, "LITE_MAX_OUTPUT_CHARS", 1600)
    if isinstance(max_chars, int) and max_chars > 0 and len(response) > max_chars:
        response = response[:max_chars].rstrip() + "..."
    add_history(store, user_input, response)
    save_store(store)
    return response
