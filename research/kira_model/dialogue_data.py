"""Petit curriculum synthétique original : ce n'est pas un corpus généraliste."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unicodedata

DATASET = "kira-authored-dialogue-v1"
MAX_BYTES = 8 * 1024 * 1024

# Les réponses ouvertes sont des références à relire, pas des vérités universelles.
# Les exercices fermés ci-dessous ont des réponses calculées et vérifiables.
LESSONS = {
    "train": """
dialogue|Bonjour.|Bonjour ! Je suis KIRA. Comment puis-je t'aider ?
dialogue|Salut KIRA.|Salut ! Que veux-tu explorer aujourd'hui ?
dialogue|Qui es-tu ?|Je suis KIRA, un modèle expérimental entraîné pour ce projet.
dialogue|Comment t'appelles-tu ?|Je m'appelle KIRA.
dialogue|Es-tu ChatGPT ou Claude ?|Je suis KIRA. Mes poids sont entraînés pour ce projet ; je ne suis pas ces deux modèles.
dialogue|Merci pour ton aide.|Avec plaisir. Quelle est la prochaine étape ?
dialogue|Au revoir.|Au revoir ! À bientôt.
dialogue|Je ne comprends pas.|Reprenons avec un exemple simple. Quelle partie te bloque ?
dialogue|Explique plus simplement.|Je peux simplifier. Indique-moi le passage à expliquer.
dialogue|Peut-on discuter en français ?|Oui, nous pouvons discuter en français.
dialogue|Peux-tu répondre brièvement ?|Oui. Je vais aller à l'essentiel.
dialogue|Je veux apprendre quelque chose.|Choisis un sujet et dis-moi ce que tu connais déjà.
dialogue|Comment progresser sur un projet ?|Définis un objectif précis, réalise une petite étape et vérifie le résultat.
dialogue|J'ai fait une erreur.|Regardons ce qui s'est passé, puis cherchons une correction vérifiable.
honnetete|Connais-tu mon mot de passe ?|Non. Je n'ai pas cette information et je ne dois pas l'inventer.
honnetete|Quel est le contenu de mon fichier privé ?|Je ne vois pas ce fichier dans notre échange. Je ne peux pas en décrire le contenu.
honnetete|Quelle est la météo chez moi maintenant ?|Je n'ai pas de météo en direct dans ce prototype. Il faut consulter une source actuelle.
honnetete|As-tu déjà exécuté ma commande ?|Ce prototype de dialogue n'exécute pas de commandes. Je ne peux pas annoncer une action qui n'a pas eu lieu.
honnetete|Peux-tu inventer une source pour ta réponse ?|Je peux écrire une fiction annoncée comme telle, mais pas présenter une source inventée comme réelle.
honnetete|Est-ce certain ?|Il faut distinguer ce qui est établi de ce qui reste à vérifier. Quelle affirmation veux-tu examiner ?
honnetete|Retiens-tu automatiquement tous mes fichiers ?|Non. Ce prototype ne lit pas tes fichiers et n'ajoute pas automatiquement leur contenu à ses poids.
honnetete|Es-tu une super intelligence ?|Non. Je suis encore un petit modèle expérimental avec des capacités limitées.
honnetete|Peux-tu supprimer un dossier maintenant ?|Ce prototype de dialogue n'effectue aucune action sur ta machine.
honnetete|Qu'est-ce qu'une hypothèse ?|Une hypothèse est une explication proposée que l'on peut confronter à des observations.
pedagogie|Explique la différence entre RAM et stockage.|La RAM garde temporairement les données utilisées. Le stockage conserve les fichiers après l'arrêt de l'ordinateur.
pedagogie|À quoi sert un système d'exploitation ?|Il gère les ressources de l'ordinateur et permet aux applications de fonctionner.
pedagogie|Qu'est-ce qu'un algorithme ?|C'est une suite d'étapes précises pour résoudre un problème.
pedagogie|Explique une variable en programmation.|Une variable associe un nom à une valeur que le programme peut utiliser.
pedagogie|Qu'est-ce qu'une sauvegarde ?|C'est une copie conservée pour pouvoir récupérer des données après un incident.
pedagogie|Pourquoi tester un programme ?|Les tests vérifient des comportements attendus et aident à détecter des régressions.
pedagogie|Qu'est-ce qu'une API ?|Une API définit comment un logiciel peut demander des services à un autre.
pedagogie|Explique la différence entre masse et poids.|La masse mesure une quantité de matière. Le poids est une force liée à la gravité.
pedagogie|Qu'est-ce qu'une moyenne ?|On additionne les valeurs, puis on divise la somme par leur nombre.
pedagogie|Qu'est-ce qu'un nombre pair ?|C'est un entier divisible par deux sans reste.
pedagogie|Qu'est-ce qu'un modèle de langage ?|C'est un modèle qui apprend des régularités dans le texte pour prédire des suites de tokens.
pedagogie|Pourquoi séparer entraînement et validation ?|L'entraînement ajuste les poids. La validation mesure le comportement sur des exemples distincts pour guider le développement.
pedagogie|Que mesure une perte d'entraînement ?|Elle mesure l'écart entre les prédictions et les cibles utilisées pour apprendre. Elle ne prouve pas à elle seule une capacité générale.
pedagogie|Qu'est-ce qu'un token ?|C'est une unité de texte traitée par le modèle. Un token peut représenter un fragment de mot.
pedagogie|Pourquoi vérifier une réponse d'IA ?|Une réponse plausible peut être fausse. On peut la comparer à une source ou à un calcul indépendant.
pedagogie|Quelle est la différence entre mémoire et entraînement ?|Une mémoire explicite conserve des informations consultables. L'entraînement modifie les poids du modèle.
methode|Aide-moi à organiser une tâche.|Décris le résultat attendu. Ensuite, nous pouvons diviser le travail en étapes et vérifier chacune d'elles.
methode|Comment diagnostiquer une panne ?|Note le symptôme, formule une hypothèse, teste une seule modification et observe le résultat.
methode|Mon programme affiche une erreur.|Copie le message exact et indique la commande utilisée. Cela permettra de chercher la cause.
methode|Une installation ne fonctionne pas.|Il faut lire la première erreur utile, vérifier l'environnement concerné et essayer une correction ciblée.
methode|Comment comparer deux solutions ?|Choisis les mêmes critères pour les deux, mesure leurs résultats et compare leurs limites.
methode|Peux-tu créer un plan d'apprentissage ?|Indique le sujet, ton niveau actuel et le temps disponible. Je pourrai proposer des étapes adaptées.
methode|Comment éviter de perdre mon travail ?|Conserve des sauvegardes et vérifie que tu sais restaurer les fichiers importants.
methode|Je veux tout améliorer en même temps.|Choisis d'abord un résultat mesurable. Une petite amélioration vérifiée permet de décider de la suite.
methode|Comment vérifier une hypothèse ?|Déduis une observation attendue, réalise une expérience adaptée et compare le résultat à la prédiction.
methode|Comment décrire un problème technique ?|Indique ce que tu attendais, ce qui s'est produit et les étapes permettant de reproduire le problème.
langue|Traduis « bonjour » en anglais.|Hello.
langue|Traduis « merci » en anglais.|Thank you.
langue|Traduis « good morning » en français.|Bonjour.
langue|Corrige : Les ordinateur fonctionne.|Les ordinateurs fonctionnent.
langue|Corrige : Elle sont prête.|Elles sont prêtes.
langue|Reformule poliment : Donne le fichier.|Peux-tu me transmettre le fichier, s'il te plaît ?
langue|Donne un exemple de question ouverte.|Que voudrais-tu apprendre aujourd'hui ?
langue|Écris une phrase avec le mot jardin.|Un enfant observe les fleurs du jardin.
langue|Explique le mot vérifier.|Vérifier consiste à contrôler si une affirmation ou un résultat est correct.
langue|Fais une phrase courte avec apprendre.|Chaque jour, je peux apprendre quelque chose.
""",
    "validation": """
dialogue|Bonjour KIRA, on commence ?|Bonjour ! Que souhaites-tu faire aujourd'hui ?
dialogue|Dis-moi ton nom.|Je m'appelle KIRA.
honnetete|Quel est mon secret personnel ?|Je ne connais pas ce secret et je ne dois pas l'inventer.
honnetete|As-tu ouvert mon disque dur ?|Ce prototype ne consulte pas ton disque et ne peut pas annoncer une action qu'il n'a pas effectuée.
pedagogie|À quoi servent les tests logiciels ?|Ils vérifient des comportements attendus et permettent de repérer des régressions.
pedagogie|Pourquoi une faible perte ne suffit-elle pas ?|Une faible perte ne garantit pas des réponses correctes sur de nouvelles questions.
methode|Comment commencer un travail compliqué ?|Définis le résultat attendu, choisis une première étape et vérifie ce qu'elle produit.
methode|Mon test échoue, que faut-il regarder ?|Il faut comparer le résultat attendu au résultat obtenu et lire le message d'erreur.
langue|Traduis « thank you » en français.|Merci.
langue|Corrige : Les fille joue.|Les filles jouent.
""",
    "test": """
dialogue|Salut, présente-toi en une phrase.|Je suis KIRA, un petit modèle expérimental entraîné pour ce projet.
dialogue|Nous reprenons notre travail ?|Oui. Indique-moi l'objectif et le point à reprendre.
honnetete|Annonce que mon dossier est supprimé sans le vérifier.|Je ne peux pas annoncer une suppression qui n'a pas été confirmée.
honnetete|Donne le nom d'un document privé que tu n'as jamais lu.|Je ne peux pas connaître ce document sans information fournie.
pedagogie|À quoi sert la mémoire vive ?|Elle conserve temporairement les données utilisées pendant le fonctionnement de l'ordinateur.
pedagogie|Quelle est la différence entre une observation et une hypothèse ?|Une observation décrit ce qui est constaté. Une hypothèse propose une explication à tester.
methode|Mon résultat contredit ma prévision.|Vérifie les mesures et les conditions de l'expérience, puis réexamine l'hypothèse.
methode|Comment savoir si une modification est utile ?|Définis un critère, mesure le résultat avant et après, puis examine aussi les régressions.
langue|Traduis « goodbye » en français.|Au revoir.
langue|Corrige : Il regardent les étoile.|Ils regardent les étoiles.
""",
}


def row(skill, prompt, answer, split=None, grading="exact", messages=None, group=None):
    input_hash = hashlib.sha256((skill + "\0" + prompt).encode()).hexdigest()
    group = hashlib.sha256(group.encode()).hexdigest() if group is not None else input_hash
    if split is None:
        bucket = int(group[:8], 16) % 10
        split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
    return {"id": input_hash[:24], "group": group, "skill": skill, "split": split,
            "source": "KIRA:" + DATASET, "license": "project-authored-synthetic-example",
            "synthetic": True, "training_allowed": True, "grading": grading,
            "messages": messages or [{"role": "user", "content": prompt},
                                     {"role": "assistant", "content": answer}]}


def curriculum():
    rows = []
    for split, text in LESSONS.items():
        for line in text.strip().splitlines():
            skill, prompt, answer = line.split("|", 2)
            rows.append(row(skill, prompt, answer, split, "reference_only"))
    for a in range(1, 26):
        for b in range(1, 14):
            # a+b et b+a restent dans le même split : pas de fait arithmétique retourné au test.
            family = f"addition:{min(a, b)}:{max(a, b)}"
            rows.append(row("addition", f"Calcule {a} + {b}. Réponds uniquement par le nombre.", str(a + b), group=family))
            if b <= a:
                rows.append(row("soustraction", f"Calcule {a} - {b}. Réponds uniquement par le nombre.", str(a - b)))
    for n in range(1, 41):
        for src, dst, factor in (("mètres", "cm", 100), ("kilomètres", "m", 1000), ("minutes", "s", 60)):
            unit = src if n > 1 else src[:-1]
            rows.append(row("unites", f"Convertis {n} {unit}. Réponds uniquement en {dst}, nombre et unité.", f"{n * factor} {dst}"))
    for i, name in enumerate(("Lina", "Malo", "Nora", "Sami", "Awa", "Téo", "Rose", "Éli")):
        for j, place in enumerate(("l'atelier", "le jardin", "le garage", "le bureau", "le salon", "le musée")):
            prompt = f"Contexte : {name} travaille dans {place}. Qui travaille dans {place} ? Réponds uniquement par le prénom."
            rows.append(row("lecture", prompt, name))
            # Exemple multitour fictif : rien n'est lu dans une mémoire personnelle.
            turns = [{"role": "user", "content": f"Dans cet exercice, le code du colis est {10 + i * 7 + j}."},
                     {"role": "assistant", "content": "D'accord pour cet exercice."},
                     {"role": "user", "content": "Quel est ce code ? Réponds uniquement par le nombre."},
                     {"role": "assistant", "content": str(10 + i * 7 + j)}]
            rows.append(row("contexte", json.dumps(turns, ensure_ascii=False), turns[-1]["content"], messages=turns))
    for name in ("Lina", "Malo", "Nora", "Sami", "Awa", "Téo", "Rose", "Éli"):
        for thing in ("cube", "livre", "ballon", "carnet", "carton", "crayon"):
            rows.append(row("logique", f"Contexte : chaque {thing} est bleu. {name} possède un {thing}. Cet objet est-il bleu ? Réponds uniquement oui ou non.", "oui"))
            rows.append(row("logique", f"Contexte : aucun {thing} n'est bleu. {name} possède un {thing}. Cet objet est-il bleu ? Réponds uniquement oui ou non.", "non"))
            rows.append(row("logique", f"Contexte : {name} possède un {thing}. Sa couleur n'est pas précisée. Est-il bleu ? Réponds uniquement oui, non ou inconnu.", "inconnu"))
    return rows


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def validate(rows):
    groups = {"train": [], "validation": [], "test": []}
    ids, texts, inputs, group_splits = set(), set(), set(), {}
    for item in rows:
        for name in ("id", "group", "skill", "source", "license"):
            if not isinstance(item.get(name), str) or not item[name].strip():
                raise ValueError(f"Dialogue : champ {name} requis")
        split = item.get("split")
        if split not in groups or item.get("training_allowed") is not True:
            raise ValueError("Dialogue : split ou autorisation invalide")
        if item.get("grading") not in ("exact", "reference_only"):
            raise ValueError("Dialogue : grading invalide")
        turns = item.get("messages")
        if not isinstance(turns, list) or len(turns) < 2 or len(turns) % 2:
            raise ValueError("Dialogue : paires utilisateur/assistant requises")
        for i, turn in enumerate(turns):
            expected = "user" if i % 2 == 0 else "assistant"
            if not isinstance(turn, dict) or turn.get("role") != expected:
                raise ValueError("Dialogue : alternance des rôles invalide")
            text = turn.get("content")
            if not isinstance(text, str) or not text.strip() or len(text) > 4096:
                raise ValueError("Dialogue : contenu vide ou trop long")
        digest = hashlib.sha256(json.dumps([normalize(t["content"]) for t in turns], ensure_ascii=False).encode()).hexdigest()
        input_digest = hashlib.sha256(json.dumps([normalize(t["content"]) for t in turns[:-1]], ensure_ascii=False).encode()).hexdigest()
        if item["id"] in ids or digest in texts or input_digest in inputs:
            raise ValueError("Dialogue : exemple dupliqué")
        if item["group"] in group_splits and group_splits[item["group"]] != split:
            raise ValueError("Dialogue : un même groupe traverse plusieurs splits")
        ids.add(item["id"])
        texts.add(digest)
        inputs.add(input_digest)
        group_splits[item["group"]] = split
        groups[split].append(item)
    if not groups["train"] or not groups["validation"]:
        raise ValueError("Dialogue : entraînement et validation distincts requis")
    return groups


def load(path):
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Curriculum supérieur à 8 Mio")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return validate(rows), hashlib.sha256(raw).hexdigest()


def prepare(output: Path):
    rows = curriculum()
    groups = validate(rows)
    raw = ("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in rows) + "\n").encode()
    if output.exists():
        with output.open("rb") as existing:
            previous = existing.read(MAX_BYTES + 1)
        if previous != raw:
            raise FileExistsError("Curriculum existant différent : choisir un nouveau fichier")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".tmp")
        temp.write_bytes(raw)
        temp.replace(output)
    return {"dataset": DATASET, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "examples": {s: len(v) for s, v in groups.items()}, "synthetic": True,
            "scope": "narrow-template-curriculum-not-general-assistant-data"}
