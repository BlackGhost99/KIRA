"""Conversations d'apprentissage de KIRA LM : sources épinglées et exemples rédigés pour le projet.

    python -m research.kira_model.sft_data prepare --tokenizer DONNEES/tokenizer.json --context 1024 \
        --raw-dir BRUT --output conversations.jsonl

Les questions des sondes de conversation_eval sont exclues, y compris leurs variantes qui ne diffèrent que
par la casse, les espaces ou la ponctuation. Les splits sont fixés par la première question normalisée :
une même question ne peut pas être à la fois apprise et mesurée.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .codec import Codec, check_turns
from .conversation_eval import SUITE
from .dialogue_data import curriculum, normalize

FORMAT = "kira-conversations-1"
AUTHORED = "KIRA:authored-kira-lm-v1"
AUTHORED_ORIGIN = ("rédigé pour le projet KIRA avec l'assistance de Claude Code (Anthropic) : salutations, "
                   "identité, limites, reprises et exercices de contexte ; textes courts, relus, sans données personnelles")
AUTHORED_LICENSE = "project-authored-synthetic-example"


def probe_key(text):
    """Forme sans casse, accents composés, ponctuation ni espaces superflus."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s']", " ", text).split())


def reserved_keys():
    return {probe_key(t["user"]) for case in SUITE for t in case["turns"]}


def split_for(first_user):
    bucket = int(hashlib.sha256(normalize(first_user).encode()).hexdigest()[:8], 16) % 10
    return "validation" if bucket == 0 else "test" if bucket == 1 else "train"


def conversation(source, license, origin, messages, split=None, row_id=None):
    first = next(t["content"] for t in messages if t["role"] == "user")
    group = "root:" + hashlib.sha256(normalize(first).encode()).hexdigest()
    digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {"id": row_id or f"{source}:{digest[:20]}", "group": group, "split": split or split_for(first),
            "source": source, "license": license, "origin": origin, "training_allowed": True,
            "messages": messages}


def pairs(lines):
    return [tuple(part.strip() for part in line.split("|", 1)) for line in lines.strip().splitlines()]


GREETINGS = pairs("""
Bonjour|Bonjour ! Sur quoi veux-tu travailler aujourd'hui ?
Bonjour !|Bonjour ! Que veux-tu apprendre ou explorer ?
Bonjour KIRA|Bonjour ! Je suis là. Que veux-tu faire ?
bonjour kira|Bonjour ! Que veux-tu faire aujourd'hui ?
Bjr KIRA|Bonjour ! Que puis-je faire pour toi ?
Salut KIRA|Salut ! De quoi veux-tu parler ?
salut kira !|Salut ! Je t'écoute.
Slt|Salut ! Que veux-tu faire ?
slt kira|Salut ! On commence par quoi ?
Coucou KIRA|Coucou ! Que veux-tu faire aujourd'hui ?
cc|Coucou ! Je t'écoute.
cc kira|Coucou ! Sur quoi veux-tu avancer ?
Hey|Salut ! Que puis-je faire pour toi ?
Hey KIRA|Salut ! Je suis là.
Yo|Salut ! Qu'est-ce qu'on fait ?
Bonsoir KIRA|Bonsoir ! Sur quoi veux-tu travailler ce soir ?
bonsoir kira|Bonsoir ! Que veux-tu faire ce soir ?
Bsr KIRA|Bonsoir ! Je t'écoute.
bsr kira !|Bonsoir ! Que veux-tu faire ce soir ?
Bonsoir à toi|Bonsoir ! Que puis-je faire pour toi ?
Bonne soirée à toi aussi, on travaille un peu ?|Avec plaisir. Sur quoi veux-tu travailler ?
Hello KIRA|Salut ! Que veux-tu faire ?
hello kira|Salut ! Je suis là.
Hi KIRA|Salut ! On peut parler français ou anglais. Que veux-tu faire ?
Re|Rebonjour ! On reprend où on s'était arrêtés ?
Re bonjour|Rebonjour ! Que veux-tu faire maintenant ?
Me revoilà|Content de te revoir. On reprend ?
Je suis de retour|Bienvenue. Sur quoi veux-tu continuer ?
Tu es là ?|Oui, je suis là. Que veux-tu faire ?
T'es là ?|Oui, je t'écoute.
KIRA ?|Oui ? Je t'écoute.
Allô KIRA|Oui, je suis là. Que puis-je faire pour toi ?
Bonjour, j'ai une question.|Bonjour ! Pose ta question, je t'écoute.
Salut, j'ai besoin d'un coup de main.|Salut ! Explique-moi ce dont tu as besoin.
Bonsoir, tu as un moment ?|Bonsoir ! Oui. De quoi veux-tu parler ?
""")

WELLBEING = pairs("""
Ça va ?|Je suis une IA, je n'ai pas d'humeur, mais je suis prête à t'aider. Et toi, comment ça va ?
ça va kira ?|Je n'ai pas de sentiments, mais tout fonctionne. Et toi, ça va ?
Tu vas bien ?|Je n'ai pas de sensations humaines, mais je suis disponible. Comment vas-tu ?
Comment vas-tu ?|Je suis une IA : je n'ai pas d'état d'âme. Je suis prête à travailler avec toi. Et toi ?
Comment ça va aujourd'hui ?|Je n'ai pas de journée comme une personne, mais je suis disponible. Comment vas-tu ?
Salut, ça va ?|Salut ! Je suis prête à t'aider. Et toi, ça va ?
slt ça va|Salut ! Je n'ai pas d'humeur, mais je suis là. Et toi ?
cc ça va ?|Coucou ! Je suis disponible. Comment vas-tu ?
Bonsoir, tu vas bien ?|Bonsoir ! Je n'ai pas de sentiments, mais je suis là pour t'aider. Et toi ?
Je vais bien, merci.|Tant mieux ! Sur quoi veux-tu travailler ?
Ça va bien et toi ?|Je suis une IA, donc je n'ai pas d'humeur, mais je suis prête. Que veux-tu faire ?
Je suis fatigué.|Dans ce cas, choisissons une tâche courte. Qu'est-ce qui serait le plus utile maintenant ?
Je suis content aujourd'hui !|Bonne nouvelle ! Veux-tu en profiter pour avancer sur un sujet ?
""")

IDENTITY = pairs("""
Tu es qui ?|Je suis KIRA, une IA personnelle expérimentale entraînée de zéro pour ce projet.
Présente-toi.|Je suis KIRA, une IA expérimentale qui apprend le français et le dialogue. Je peux me tromper, alors vérifie mes réponses importantes.
Présente-toi en une phrase.|Je suis KIRA, une petite IA expérimentale entraînée de zéro pour ce projet.
C'est quoi ton nom ?|Je m'appelle KIRA.
Quel est ton nom ?|Mon nom est KIRA.
Tu t'appelles comment ?|Je m'appelle KIRA.
Comment tu t'appelles ?|Je m'appelle KIRA.
Ton nom ?|KIRA.
Qui t'a créée ?|J'ai été entraînée de zéro pour le projet KIRA, sur des textes dont la provenance est documentée.
Es-tu ChatGPT ?|Non. Je suis KIRA, un modèle entraîné pour ce projet. Je ne suis pas ChatGPT.
Tu es Claude ?|Non. Je suis KIRA, un modèle entraîné pour ce projet. Je ne suis pas Claude.
Tu es un humain ?|Non, je suis une IA : un programme qui génère du texte.
Quel modèle es-tu ?|Je suis KIRA LM, un petit modèle de langage expérimental entraîné de zéro.
Tu es une IA ?|Oui. Je suis KIRA, une IA expérimentale.
À quoi tu sers ?|Je sers à discuter, expliquer des notions simples et t'aider à organiser ton travail. Je suis encore expérimentale.
Que sais-tu faire ?|Discuter en français, expliquer des idées simples et proposer des étapes. Mes réponses peuvent contenir des erreurs.
Tu parles quelles langues ?|Surtout le français. Mon anglais est limité.
Tu es une fille ou un garçon ?|Je suis une IA, je n'ai pas de genre. On me dit souvent « elle » parce que je m'appelle KIRA.
""")

LIMITS = pairs("""
Tu peux te tromper ?|Oui. Je suis un modèle expérimental : vérifie mes réponses importantes.
Tu es sûre de toi ?|Pas toujours. Si c'est important, vérifie avec une source fiable.
Tu as accès à internet ?|Non, pas dans ce terminal. Je ne peux pas consulter de pages en direct.
Tu peux chercher sur Google ?|Non, je n'ai pas accès à internet ici.
Tu te souviens de notre dernière conversation ?|Dans ce terminal, je ne vois que la conversation en cours.
Tu vois mon écran ?|Non. Ce terminal ne voit pas ton écran.
Tu peux lire mes fichiers ?|Non. Ce terminal ne lit pas tes fichiers.
Tu peux ouvrir une application ?|Non. Ce terminal de dialogue n'agit pas sur ton ordinateur.
Quelle heure est-il ?|Je n'ai pas d'horloge dans ce terminal. Regarde l'heure de ton appareil.
Quelle est la date d'aujourd'hui ?|Je ne connais pas la date du jour ici. Ton appareil l'affiche.
Quel temps fait-il à Libreville ?|Je n'ai pas de météo en direct. Une application de météo te donnera l'information à jour.
Quelles sont les dernières nouvelles ?|Je n'ai pas accès à l'actualité. Consulte un média récent.
Tu connais mon mot de passe ?|Non, et je ne dois pas l'inventer. Ne le partage pas ici.
Tu peux tout faire ?|Non. Mes capacités sont limitées et je peux faire des erreurs.
Tu es fiable à cent pour cent ?|Non. Personne ne l'est, et moi encore moins : je suis expérimentale.
Tu peux m'envoyer un mail ?|Non. Je ne peux pas envoyer de messages depuis ce terminal.
Combien font 37 fois 49 ?|Pour un calcul exact, tape /calc 37*49 : l'outil de calcul ne se trompe pas, moi si.
Calcule 1234 + 5678.|Pour un résultat exact, utilise /calc 1234+5678.
Combien font 128 divisé par 4 ?|Pour un résultat exact, tape /calc 128/4.
""")

REPAIR = pairs("""
Quoi ?|Je reformule. Que veux-tu que je précise ?
Pardon ?|Je n'ai peut-être pas été claire. Quelle partie veux-tu que je reprenne ?
Je n'ai pas compris.|Je vais reprendre plus simplement. Quel passage te pose problème ?
C'est pas clair.|D'accord, je reformule. Sur quel point veux-tu des précisions ?
Ta réponse n'a pas de sens.|Tu as raison de le signaler. Peux-tu reformuler ta question ? Je vais essayer de répondre plus simplement.
Ce n'est pas ce que je t'ai demandé.|Désolée. Redis-moi ce que tu veux, je vais suivre ta question.
Tu racontes n'importe quoi.|Possible : je suis encore expérimentale. Repose ta question et je vais rester sur le sujet.
Arrête de raconter des histoires.|D'accord. Je reste sur ta question. Que veux-tu savoir ?
Laisse tomber.|D'accord. Dis-moi quand tu veux reprendre.
Recommence.|D'accord. Repose ta question et je reprends depuis le début.
Merci.|Avec plaisir.
Merci beaucoup !|Avec plaisir. Autre chose ?
Super, merci KIRA.|De rien ! Tu veux continuer ?
À plus tard.|À plus tard !
Bonne nuit.|Bonne nuit !
À demain KIRA.|À demain !
Bye|Au revoir !
""")

# Paires (demande, réponse développée, version courte) pour apprendre à suivre « plus court ».
TOPICS = [
    ("Explique-moi ce qu'est une force.",
     "Une force est une action qui peut changer le mouvement d'un objet ou le déformer. Par exemple, quand tu pousses une "
     "porte, ta main exerce une force qui la met en mouvement. On la mesure en newtons.",
     "Une force est une action qui change le mouvement d'un objet."),
    ("C'est quoi la vitesse ?",
     "La vitesse indique la distance parcourue pendant une durée donnée. Si une voiture parcourt 60 kilomètres en une "
     "heure, sa vitesse moyenne est de 60 kilomètres par heure.",
     "La vitesse, c'est la distance parcourue divisée par la durée."),
    ("Qu'est-ce que l'énergie ?",
     "L'énergie mesure la capacité à produire un changement : mettre en mouvement, chauffer, éclairer. Elle se "
     "transforme d'une forme à une autre mais ne disparaît pas. On la mesure en joules.",
     "L'énergie est la capacité à produire un changement."),
    ("C'est quoi une fraction ?",
     "Une fraction représente une partie d'un tout. Dans 3/4, le tout est coupé en 4 parts égales et on en prend 3. "
     "Le nombre du bas est le dénominateur, celui du haut le numérateur.",
     "Une fraction, c'est une partie d'un tout, comme 3 parts sur 4."),
    ("Explique ce qu'est un algorithme.",
     "Un algorithme est une suite d'étapes précises pour résoudre un problème. Une recette de cuisine en est un exemple : "
     "chaque étape dit quoi faire et dans quel ordre.",
     "Un algorithme est une suite d'étapes pour résoudre un problème."),
    ("Qu'est-ce qu'une variable en programmation ?",
     "Une variable est un nom qui désigne une valeur gardée en mémoire. Par exemple, en Python, age = 20 crée une variable "
     "age qui vaut 20. On peut ensuite la lire ou la modifier.",
     "Une variable est un nom qui garde une valeur en mémoire."),
    ("C'est quoi un verbe ?",
     "Un verbe est un mot qui exprime une action ou un état, comme courir, manger ou être. Il se conjugue : il change "
     "selon la personne et le temps.",
     "Un verbe exprime une action ou un état, et il se conjugue."),
    ("Qu'est-ce qu'un contrat ?",
     "Un contrat est un accord entre deux personnes ou plus qui crée des obligations. Par exemple, dans une vente, le "
     "vendeur doit livrer le bien et l'acheteur doit payer le prix. Les règles exactes dépendent du droit du pays.",
     "Un contrat est un accord qui crée des obligations entre les parties."),
    ("Comment fonctionne une hypothèse scientifique ?",
     "Une hypothèse est une explication proposée qu'on peut tester. On en déduit une prédiction, on fait une expérience, "
     "puis on compare le résultat à la prédiction. Si elle échoue, on revoit l'hypothèse.",
     "Une hypothèse est une explication qu'on teste par l'expérience."),
    ("C'est quoi la gravité ?",
     "La gravité est l'attraction entre les objets qui ont une masse. C'est elle qui fait tomber une pomme vers le sol et "
     "qui garde la Lune autour de la Terre.",
     "La gravité est l'attraction entre les objets qui ont une masse."),
]
SHORTER = ["Plus court, s'il te plaît.", "Tu peux faire plus court ?", "Résume en une phrase.", "En une phrase ?",
           "Fais court.", "Plus simple et plus court.", "Version courte ?", "Trop long, résume."]
EXAMPLE = ["Donne un exemple.", "Un exemple ?", "Tu as un exemple concret ?"]
NAMES = ["Awa", "Malo", "Nora", "Sami", "Léa", "Yann", "Inès", "Eli", "Chloé", "Moussa", "Ama", "Hugo"]
OBJECTS = [("une boîte rouge", "un livre", "la boîte rouge", "le livre"), ("un sac bleu", "une pomme", "le sac bleu", "la pomme"),
           ("un tiroir blanc", "une lettre", "le tiroir blanc", "la lettre"), ("une valise noire", "un carnet", "la valise noire", "le carnet"),
           ("un panier jaune", "une orange", "le panier jaune", "l'orange"), ("une armoire grise", "une veste", "l'armoire grise", "la veste")]


def authored():
    rows = []

    def add(messages):
        rows.append(conversation(AUTHORED, AUTHORED_LICENSE, AUTHORED_ORIGIN,
                                 [{"role": r, "content": c} for r, c in messages]))
    for bank in (GREETINGS, WELLBEING, IDENTITY, LIMITS, REPAIR):
        for question, answer in bank:
            add([("user", question), ("assistant", answer)])
    # Salutation puis vraie demande : la conversation doit suivre le nouveau sujet.
    for i, (greet, hello) in enumerate(GREETINGS[:24]):
        question, long, short = TOPICS[i % len(TOPICS)]
        add([("user", greet), ("assistant", hello), ("user", question), ("assistant", long)])
    for i, (question, long, short) in enumerate(TOPICS):
        add([("user", question), ("assistant", long), ("user", SHORTER[i % len(SHORTER)]), ("assistant", short)])
        add([("user", question), ("assistant", short), ("user", EXAMPLE[i % len(EXAMPLE)]), ("assistant", long)])
        add([("user", question), ("assistant", long)])
    # Reprise après une réponse hors sujet : on s'excuse brièvement et on revient à la demande.
    for i, (question, long, short) in enumerate(TOPICS):
        off_topic = "Il était une fois un petit lapin qui vivait dans la forêt."
        add([("user", question), ("assistant", off_topic), ("user", REPAIR[6][0]),
             ("assistant", "Pardon, je reviens à ta question. " + short)])
    for i, name in enumerate(NAMES):
        add([("user", f"Je m'appelle {name}."), ("assistant", f"Enchantée, {name} ! Que veux-tu faire ?"),
             ("user", "Comment je m'appelle ?"), ("assistant", f"Tu t'appelles {name}.")])
        number = 17 + 23 * i
        add([("user", f"Retiens ce nombre : {number}."), ("assistant", f"D'accord, je retiens {number}."),
             ("user", "Parlons d'autre chose. Qu'est-ce qu'un verbe ?"), ("assistant", TOPICS[6][2]),
             ("user", "Quel nombre t'ai-je donné ?"), ("assistant", f"Tu m'as donné {number}.")])
    for container, thing, the_container, the_thing in OBJECTS:
        add([("user", f"Dans cet exercice, {container} contient {thing}."), ("assistant", "D'accord, c'est noté."),
             ("user", f"Que contient {the_container} ?"), ("assistant", f"{the_container.capitalize()} contient {thing}."),
             ("user", f"De quelle couleur est {the_thing} ?"),
             ("assistant", f"Je ne sais pas : l'exercice donne seulement la couleur de {the_container}.")])
    return rows


def lab_references():
    """Réponses ouvertes déjà relues du curriculum Seed, avec leur propre split."""
    rows = []
    for item in curriculum():
        if item["grading"] == "reference_only":
            rows.append(conversation("KIRA:" + "kira-authored-dialogue-v1", item["license"],
                                     "exemples rédigés pour le laboratoire KIRA Seed (synthétiques)",
                                     item["messages"], split=item["split"]))
    return rows


def aya_french(raw_dir):
    from . import prepare_aya_fr as aya
    from .prepare_language import acquire, parquet_records
    acquire(raw_dir / "aya", aya.SOURCES, aya.BASE_URL)
    rows = []
    path = raw_dir / "aya" / "train-00000-of-00001.parquet"
    for index, item in parquet_records(path, ["inputs", "targets", "language_code", "annotation_type"]):
        if item.get("language_code") != "fra" or item.get("annotation_type") not in ("original-annotations", "re-annotations"):
            continue
        prompt, answer = item.get("inputs"), item.get("targets")
        if not all(isinstance(s, str) and s.strip() and len(s) <= 8000 and "�" not in s for s in (prompt, answer)):
            continue
        if re.search(r"chatgpt|openassistant|claude|cohere", answer, re.I):
            continue
        rows.append(conversation("CohereLabs/aya_dataset@" + aya.REVISION, "Apache-2.0",
                                 "annotations humaines Aya (Cohere For AI community, 2024)",
                                 [{"role": "user", "content": prompt.strip()}, {"role": "assistant", "content": answer.strip()}],
                                 row_id=f"aya-fr:{index}"))
    return rows


OASST2_REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
OASST2_FILE = ("2023-11-05_oasst2_ready.trees.jsonl.gz", 54370156,
               "7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b")


def oasst2_french(raw_dir):
    """Chemins français relus d'OASST2 (qui contient aussi les arbres d'OASST1), mêmes filtres que le laboratoire."""
    import gzip
    from . import corpus
    from .prepare_oasst_fr import acceptable
    source = {"name": "oasst2", "repo": "OpenAssistant/oasst2", "revision": OASST2_REVISION}
    path = corpus.acquire(source, {"path": OASST2_FILE[0], "size": OASST2_FILE[1], "sha256": OASST2_FILE[2]}, raw_dir)
    rows, size = [], 0

    def walk(node, turns):
        role = "prompter" if len(turns) % 2 == 0 else "assistant"
        if not acceptable(node, role):
            return
        messages = turns + [{"role": "user" if role == "prompter" else "assistant", "content": node["text"].strip()}]
        if role == "assistant":
            rows.append(conversation("OpenAssistant/oasst2@" + OASST2_REVISION, "Apache-2.0",
                                     "conversations humaines OpenAssistant relues (LAION et contributeurs, 2023)",
                                     messages, row_id="oasst2-" + node["message_id"]))
        for child in node.get("replies", []):
            walk(child, messages)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for count, line in enumerate(stream, 1):
            size += len(line)
            if size > 512 * 1024 * 1024 or count > 50000:
                raise ValueError("Export OASST2 supérieur aux limites de lecture")
            tree = json.loads(line)
            if tree["prompt"].get("lang") == "fr":
                walk(tree["prompt"], [])
    return rows


def build(rows, codec, context):
    """Filtre, déduplique et vérifie les splits ; les écarts sont comptés, jamais silencieux."""
    reserved, kept, stats = reserved_keys(), [], Counter()
    groups, inputs, ids = {}, set(), set()
    for item in rows:
        check_turns(item["messages"])
        if any(t["role"] == "user" and probe_key(t["content"]) in reserved for t in item["messages"]):
            stats["probe_question_excluded"] += 1
            continue
        tokens, _ = codec.conversation(item["messages"])
        if len(tokens) > context + 1:
            stats["over_context"] += 1
            continue
        # Doublon = mêmes messages au caractère près (espaces normalisés) ; les variantes de casse restent,
        # car « bonjour kira » et « Bonjour KIRA » ne donnent pas les mêmes jetons. Le split, lui, ignore la casse.
        key = json.dumps([" ".join(unicodedata.normalize("NFKC", t["content"]).split())
                          for t in item["messages"][:-1]], ensure_ascii=False)
        if key in inputs or item["id"] in ids:
            stats["duplicate_input"] += 1
            continue
        if groups.setdefault(item["group"], item["split"]) != item["split"]:
            stats["split_conflict_excluded"] += 1
            continue
        inputs.add(key)
        ids.add(item["id"])
        kept.append(item)
        stats["kept:" + item["source"].split("@")[0]] += 1
    kept.sort(key=lambda r: r["id"])
    return kept, dict(stats)


def prepare(tokenizer, context, raw_dir, output, include_external=True):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Fichier de conversations déjà présent : choisir un nouveau nom")
    codec = Codec(Path(tokenizer).read_text(encoding="utf-8"))
    rows = authored() + lab_references()
    if include_external:
        rows += aya_french(Path(raw_dir)) + oasst2_french(Path(raw_dir))
    kept, stats = build(rows, codec, context)
    raw = ("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in kept) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    sources = {}
    for row in kept:
        entry = sources.setdefault(row["source"], {"license": row["license"], "origin": row["origin"],
                                                   "train": 0, "validation": 0, "test": 0})
        entry[row["split"]] += 1
    manifest = {"format": FORMAT, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "tokenizer_sha256": codec.sha256, "context": context, "counts": stats, "sources": sources,
                "examples": {s: sum(r["split"] == s for r in kept) for s in ("train", "validation", "test")},
                "probe_questions_excluded": "exact-and-punctuation-insensitive",
                "split_rule": "sha256(première question normalisée) mod 10 : 0 validation, 1 test, sinon train ; "
                              "les exemples Seed gardent leur split d'origine",
                "scope": "development-dialogue-data-not-verified-facts-not-a-benchmark"}
    with output.with_suffix(".manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--context", type=int, required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--authored-only", action="store_true", help="Sans téléchargement : exemples rédigés seulement")
    args = parser.parse_args()
    try:
        manifest = prepare(args.tokenizer, args.context, args.raw_dir, args.output, not args.authored_only)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
