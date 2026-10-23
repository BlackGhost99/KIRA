# KIRA

**Une IA personnelle avec son propre modèle, un cœur installé en arrière-plan et un moteur cloud.**

## Le but : une IA « mix ChatGPT–Claude »

L'ambition de KIRA est de réunir les qualités que nous recherchons chez ChatGPT et
Claude : dialogue naturel, raisonnement rigoureux, programmation, analyse de documents,
explications claires et collaboration sur des projets qui durent.

Nous voulons aller vers une assistante qui connaît le contexte de son propriétaire,
retrouve ses informations, utilise des outils vérifiables et peut agir sur ses appareils
avec les autorisations prévues. Elle doit aussi reconnaître une information manquante,
rester cohérente au fil d'un échange et permettre de corriger sa mémoire.

Cette ambition guide le développement d'un **modèle KIRA indépendant, initialisé de zéro**.
Les expériences actuelles utilisent nos propres poids et un tokenizer appris pour KIRA,
avec des méthodes publiques, PyTorch et des corpus dont la provenance est documentée.
L'expression « mix ChatGPT–Claude » décrit l'objectif de qualité et de comportement ;
elle ne signifie pas que leurs poids ont été récupérés ou fusionnés.

**Ce niveau reste un objectif : le modèle maison actuel ne sait pas encore tenir une
conversation fiable.** Le dépôt conserve ses progrès, ses échecs et les mesures qui
permettent de décider de la suite.

## Priorités du projet

1. **Le core installé**, discret au quotidien : le mode « astral » de KIRA, avec un état
   consultable, un journal et une possibilité de pause.
2. **Le moteur cloud**, pour le calcul, la mémoire et l'orchestration des outils.
3. **Le propre modèle de KIRA**, développé et évalué progressivement avant son intégration.
4. **L'interface graphique**, utile pour discuter et contrôler l'assistante, mais secondaire.

Le projet vise une assistante installée et persistante. L'interface web existante sert
aujourd'hui de panneau de contrôle ; créer un site public n'est pas le but du projet.

## Où nous en sommes — 23 octobre 2026

Le dépôt rassemble deux chantiers : le système d'assistance et le laboratoire du modèle.
Le système contient déjà du code pour la mémoire, les projets, les outils et les machines.
Le laboratoire cherche encore à obtenir un dialogue pertinent et stable avec nos propres poids.

| Élément | Présent dans le code | État et limites |
|---|---|---|
| **KIRA Core** | Appairage, dossiers autorisés, actions sous permissions, corbeille, pause et démarrage automatique | Contrats de fichiers/périmètre testés sur Linux, Windows et macOS ; installation automatique, écran et contrôle matériel restent à essayer sur les appareils réels. |
| **Moteur serveur** | Conversation, outils, tâches de fond persistantes, reprise des travaux compatibles et journaux | Le dialogue du serveur utilise encore les fournisseurs configurés ; le modèle maison n'y est pas intégré. |
| **Mémoire V3** | Souvenirs versionnés, provenance, correction, archives, identité persistante et recherche hybride préparée | La recherche sémantique exige un service d'embeddings et PostgreSQL/pgvector configurés ; les tests simulés ne valident pas sa pertinence réelle. |
| **Sessions et routeur** | Cookies HttpOnly révocables, modes QUALITY/ECONOMY/PRIVATE, bascule entre fournisseurs compatibles | Les appels réels dépendent des clés, modèles, permissions et quotas du compte configuré. |
| **Continuité V4** | Projets, tâches, documents/OCR, cartes d'apprentissage, sauvegarde et restauration | Code et contrats testés ; les parcours réels dans l'environnement du propriétaire restent à valider. |
| **Interface** | Panneau web installable, projets, mémoire, accords et dictée manuelle | Les tests DOM ne remplacent pas un essai visuel, vocal ou mobile réel. |
| **Modèle KIRA Seed 0.4** | Préparation des corpus, entraînement CPU, sauvegarde/reprise et dialogue local | Prototype de recherche : réponses hors sujet, répétitions et mauvaise gestion du contexte. Non qualifié pour servir d'assistante. |
| **KIRA LM 0.5 (Kaggle)** | Modèle type Llama de ≈ 100 M de paramètres, corpus français épinglé (≈ 5,4 G jetons), entraînement GPU reprenable, notebooks Kaggle | Testé sur CPU et en simulation des notebooks ; pas encore entraîné sur GPU. Vitesse et qualité à mesurer. Voir le [guide Kaggle](docs/kira-kaggle.md). |

Les fournisseurs externes du serveur et les poids du laboratoire sont des mécanismes
distincts. Un appel à Groq, OpenAI ou Anthropic reste un appel à ce fournisseur.
La mémoire enregistrée en base n'est pas un réentraînement du modèle KIRA.

La publication du code ne certifie pas l'état du service hébergé. La configuration réelle,
la connexion à la base, les fournisseurs et les appareils doivent être vérifiés sur
l'installation utilisée. Le laboratoire fonctionne sans clé d'API et sans carte graphique dédiée.

## Le laboratoire : ce qui a été fait

Le dossier [research/kira_model](research/kira_model) contient un petit Transformer causal
créé pour apprendre depuis zéro sur CPU. La chaîne a progressé du codage par octets à un
vocabulaire BPE, puis à une phase de français simple et à une adaptation au dialogue.

Le pilote 0.4 utilise **1 088 512 paramètres avec les rôles de dialogue**, un contexte de
256 tokens et des poids issus de notre propre chaîne. La phase linguistique supplémentaire
compte 10 000 étapes, suivies de 4 000 étapes d'adaptation au dialogue. Ces nombres décrivent
des phases séparées de l'expérience, pas un score d'intelligence.

Les données comprennent des textes littéraires français, un échantillon d'histoires
LeCarnet et des conversations françaises issues d'Aya, d'OpenAssistant et d'exemples
originaux KIRA. Les attributions, licences, origines synthétiques, empreintes et séparations
train/validation/test sont consignées dans les préparateurs et les rapports.
Les poids préentraînés des modèles de ces auteurs ne sont pas utilisés.

### Résultat actuel : progrès mesurables, dialogue insuffisant

Sur quelques formulations, le pilote répond correctement :

| Message | Réponse observée dans la session initiale |
|---|---|
| `bjr` | Bonjour ! Que souhaites-tu faire ? |
| `Comment t'appelles tu ?` | Je m'appelle KIRA. |

Mais une variante courante comme **`bsr` déclenche une histoire de Lulu**, puis le modèle
continue ce récit malgré les messages suivants. Cet échec a été reproduit mot pour mot
avec le fichier distribué. La casse, la formulation et son propre historique peuvent
faire dérailler ses réponses. Les demandes de projet et le rappel du contexte échouent aussi.

Deux prolongations de 2 000 étapes ont été comparées, avec et sans répétition des histoires.
Elles améliorent certaines mesures, mais restent incohérentes et dégradent d'autres réponses.
**Aucun de ces essais supplémentaires n'est proposé comme modèle corrigé.**

Les anciens petits exercices fermés sont passés de 8/30 pour le pilote 0.3 à 0/30 pour
le pilote 0.4. L'outil explicite `/calc` continue de calculer exactement : c'est un outil
arithmétique du terminal, distinct des capacités apprises par le modèle.

La vérification comprend désormais **30 tours de conversation de développement**, avec
variantes de salutations, suivi du contexte et reprise après une mauvaise réponse.
Les vraies sorties du modèle alimentent les tours suivants. Ces sondes connues et leur
relecture par Codex ne constituent pas une évaluation humaine indépendante.

Rapports et détails :

- [Direction générale et laboratoire depuis zéro](docs/kira-own-model.md)
- [Expérience BPE sur CPU](docs/kira-seed-bpe.md)
- [Apprentissage du dialogue](docs/kira-seed-dialogue.md)
- [Expérience 0.4 et limites constatées](docs/kira-language-0.4.md)
- [KIRA LM 0.5 : entraîner sur les GPU gratuits de Kaggle](docs/kira-kaggle.md)
- [Reproduction du dialogue « bsr »](research/kira_model/results/dialogue-v04-owner-bsr-failure.json)
- [Comparaison des deux prolongations](research/kira_model/results/dialogue-v04-replay-study.json)

Les corpus téléchargés, les dossiers de runs et les checkpoints d'entraînement restent
hors de Git. Le dépôt contient le code, les petites données de test et les rapports.
Publier cette version ne remplace pas automatiquement les poids présents sur un PC.

## Comment les éléments travaillent ensemble

| Élément | Rôle |
|---|---|
| Modèle | Produire et interpréter du langage ; modèle KIRA encore expérimental, fournisseurs configurés utilisés par le serveur actuel. |
| Service cloud | Orchestrer la conversation, les projets, la mémoire, les tâches et les outils. |
| Mémoire | Conserver les informations avec leur origine, leurs corrections et leur statut. |
| Core installé | Appliquer les actions autorisées dans le périmètre de la machine. |
| Interface | Permettre au propriétaire de discuter, vérifier, approuver, corriger et arrêter. |

Le core initie les échanges avec le serveur. Ses pouvoirs et dossiers sont plafonnés
sur la machine. Les permissions peuvent exiger un accord ou interdire une action ;
la pause locale reste disponible. Le terminal expérimental du modèle ne dispose pas
de ces pouvoirs et ne contrôle pas l'ordinateur.

## Essayer et développer

### Laboratoire du modèle sur Windows / PowerShell

Depuis la racine du dépôt, créer un environnement séparé du serveur :

```powershell
python -m venv .venv-model
$kiraModelPython = (Resolve-Path ".venv-model\Scripts\python.exe").Path

& $kiraModelPython -m pip install --index-url https://download.pytorch.org/whl/cpu -r research/kira_model/requirements.txt
& $kiraModelPython -m pip install --no-deps --only-binary=:all: -r research/kira_model/requirements-bpe.txt
& $kiraModelPython -m unittest discover -s research/kira_model/tests -v
```

Le paquet Tokenizers est installé sans les dépendances Hub ; le laboratoire n'utilise
ni leurs API ni PyYAML pour entraîner ou converser. Les versions épinglées sont dans
les deux fichiers de dépendances. Sous Linux/macOS, utiliser le Python de
`.venv-model/bin/python` après création de l'environnement.

Pour charger des poids de dialogue déjà obtenus, remplacer le chemin ci-dessous par
celui de son propre checkpoint KIRA compatible :

```powershell
$kiraCheckpoint = "C:\chemin\vers\vos-poids-kira.pt"
& $kiraModelPython -m research.kira_model.dialogue chat --checkpoint $kiraCheckpoint
```

Commandes du terminal : `/reset` pour vider l'historique, `/quit` pour sortir,
`/calc expression` pour le calcul exact. Vider l'historique ne corrige pas les poids.
Les guides du laboratoire expliquent la préparation, l'entraînement et la reprise
sans écraser un run existant. Les temps publiés dépendent de la machine qui les mesure.

### Système serveur et interface

Les dépendances du serveur sont dans [requirements.txt](requirements.txt) et ses
réglages dans [.env.example](.env.example). Utiliser un autre environnement virtuel
que celui du modèle, installer ces dépendances puis copier `.env.example` vers `.env`
pour le développement local. L'application lit ce fichier ; les variables déjà définies
dans l'environnement gardent la priorité.

Après configuration de l'accès et d'un fournisseur, lancer depuis cet environnement :

```bash
python -m uvicorn app.main:app --reload
```

Pour un développement en HTTP local, définir explicitement `COOKIE_SECURE=false` ;
conserver les cookies sécurisés pour l'installation HTTPS. La configuration des sessions,
des modes et des fournisseurs est détaillée dans le [guide du routeur](docs/router-auth-v3.md).
Les secrets restent dans l'environnement de l'installation, hors du dépôt.

Une démonstration locale permet d'examiner l'interface sans clé d'API :

```bash
python scripts/dev_demo.py
```

Elle utilise **des réponses simulées**, une base temporaire et une machine de démonstration
isolée. L'adresse affichée est `http://127.0.0.1:8765`, avec le mot de passe `demo`.
Cette démonstration ne mesure pas les capacités du modèle KIRA.

Pour relier un appareil au système serveur : **Mes machines → Ajouter une machine**,
puis suivre les commandes d'appairage générées. L'installation discrète et les capacités
matérielles doivent être validées sur l'appareil concerné.

## Validation

La CI sépare les contrôles du système et ceux du modèle :

- **Système** : tests Python, base PostgreSQL/pgvector jetable, sessions, continuité,
  OCR, parcours DOM et contrats du core sur Linux, Windows et macOS.
- **Modèle** : 78 tests techniques, exécutés sur Linux/Python 3.12 et Windows/Python 3.15,
  plus des expériences CPU et les générations conservées pour relecture.
- **Qualité du dialogue** : conversations avec les sorties réelles, variantes et
  régressions documentées ; les pertes et le nombre de tests techniques ne la certifient pas.

Les workflows système et modèle du commit `9f96f641f44a30e92521135fbd9835b1f5902cb2`
ont réussi. Les tests utilisant des services simulés ne valident pas les clés, les quotas,
les réponses des fournisseurs réels ou l'installation matérielle.

## Prochaines étapes

1. Lancer KIRA LM 0.5 sur Kaggle en suivant le [guide](docs/kira-kaggle.md) : vérifier le compte par téléphone,
   préparer les données, mesurer la vitesse réelle sur deux T4, puis enchaîner les sessions de préentraînement.
2. Relire les textes générés et les 30 sondes de conversation après l'adaptation au dialogue, en conservant
   les échecs et les régressions, puis évaluer sur des conversations séparées des cas de développement.
3. Élargir les données de dialogue françaises après examen de leurs conditions (traductions et textes générés
   par d'autres IA compris), puis apprendre des appels d'outils vérifiables.
4. Qualifier un modèle réellement utile avant de le relier au moteur cloud et à sa mémoire (export prévu vers
   llama.cpp/Ollama, que le routeur connaît déjà).
5. Valider le core discret sur les appareils réels : démarrage, pause, permissions,
   résultat des actions et continuité après redémarrage.
6. Améliorer ensuite le confort graphique, la voix et la vision en fonction des usages.

## Repères dans le dépôt

| Chemin | Contenu |
|---|---|
| [research/kira_model/](research/kira_model) | Modèle, données de test, préparation, entraînement, évaluations et rapports. |
| [core/](core) | Agent installé sur les machines. |
| [app/](app) | Serveur, mémoire, sessions, routeur, projets, outils et tâches. |
| [web/](web) | Interface de discussion et de contrôle. |
| [tests/](tests) | Tests du système et du core. |
| [scripts/](scripts) | Démonstration, indexation mémoire et sauvegardes personnelles. |
| [docs/](docs) | Conception, commandes, résultats et limites. |
| [supabase/](supabase) | Migration vectorielle et fonction d'embeddings optionnelle. |
| [render.yaml](render.yaml) | Configuration d'hébergement existante. |
| [legacy/](legacy) | Ancienne version conservée pour référence. |

Documentation du système : [mémoire V3](docs/memory-v3.md),
[sessions et fournisseurs](docs/router-auth-v3.md),
[continuité, documents et sauvegardes](docs/continuity-v4.md).
