# KIRA Seed Dialogue 0.3

Le propriétaire veut développer une assistante qui réunisse les qualités qu'il
apprécie chez ChatGPT et Claude. La cible reste un core installé en arrière-plan,
un moteur cloud et une interface secondaire. L'étape 0.3 fournit une chaîne
d'apprentissage du dialogue à partir de **nos propres poids KIRA** et des mesures
qui permettent de refuser une régression. Elle n'établit pas une capacité générale
comparable aux assistants de référence.

## Point de départ mesuré

Le PC disponible est un i5-2520M avec 8 Go, sans GPU dédié. Le propriétaire a achevé
4 000 étapes BPE avec 1 088 256 paramètres, une validation littéraire de
4,8551117276 par token BPE (2,5036801918 bits par octet UTF-8) et
0,289808 seconde par étape. Ses générations restent incohérentes.
Ce modèle a appris essentiellement deux ouvrages. Une baisse de perte ou un
nouveau tokenizer ne suffisent pas à démontrer compréhension et raisonnement.

Le modèle initial est conservé. La nouvelle chaîne charge son vocabulaire et tous
ses poids, ajoute deux marqueurs de rôle initialisés aléatoirement, puis poursuit
l'apprentissage. Le profil du propriétaire passe à **1 088 512 paramètres**.
Le tokenizer n'est pas réentraîné. Aucun poids ou appel à un modèle tiers n'est
utilisé pour apprendre ou noter les réponses.

## Ce qui est ajouté

- Conversations avec rôles utilisateur/assistant séparés par des identifiants
  virtuels hors du vocabulaire texte. Écrire littéralement une balise ne crée pas
  un nouveau rôle.
- Perte calculée uniquement sur les réponses de l'assistante et leur fin de tour
  pour les exemples de dialogue. Les questions et le padding sont masqués.
- Lots de longueur réelle pour réduire le calcul inutile sur CPU ; équilibrage
  des familles d'exercices, warmup, décroissance cosinus et écrêtage du gradient.
- Répétition du corpus français antérieur sur 25 % des étapes lorsqu'il est fourni.
  Ces étapes gardent leur objectif de prédiction du prochain token, distinct de
  l'objectif de dialogue. Les tokens vus sont comptés séparément.
- Meilleurs poids choisis sur la perte de validation du dialogue, à condition que
  la perte française n'augmente pas de plus de **0,35 nat par token** par rapport
  au départ. Ce seuil expérimental est fixé avant le pilote et reste configurable.
- `training.pt` atomique contient poids courants, meilleurs poids, optimiseur,
  RNG, tokenizer, signatures et compteurs. `best.pt` est autonome. `last.pt`
  conserve le dernier état ; il n'est pas supposé être le meilleur.
- Terminal de conversation avec contexte borné, `/reset`, `/quit` et outil de
  calcul explicite `/calc`. Cet outil utilise des fractions exactes et une
  grammaire arithmétique limitée ; il n'exécute aucun code Python fourni.

La mémoire d'une conversation dans ce terminal est temporaire et limitée au
contexte. Ce n'est pas la mémoire persistante du core existant. Le terminal ne
déclenche aucune action sur les appareils et ne remplace pas le moteur cloud.
Le résultat de `/calc` vient d'un outil déterministe, pas d'une compétence apprise
dans les poids du modèle.

## Curriculum et portée de l'évaluation

`dialogue_data.py` écrit localement un curriculum synthétique original :
**1 012 exemples**, dont **808 d'entraînement, 99 de validation et 105 de test**.
Il couvre de petits exercices de dialogue, honnêteté, méthode, vocabulaire,
addition, soustraction, unités, lecture de contexte et logique simple.
Les calculs sont générés par des opérations Python vérifiables. Les réponses
ouvertes sont des références à relire et ne sont pas notées par égalité comme une
mesure de qualité sémantique. Le corpus ne contient aucune conversation privée.

Les groupes et les entrées normalisées ne peuvent pas traverser les splits.
Deux additions commutées restent dans le même groupe. Les templates restent
proches entre splits : il s'agit d'un diagnostic pédagogique étroit, pas d'un
benchmark indépendant de français général ou d'intelligence. Le test réservé
n'est ni encodé pendant l'entraînement, ni utilisé pour choisir les poids.

`evaluate` utilise par défaut la **validation** : sélection stable par identifiant,
répartie entre familles, décodage glouton, égalité normalisée pour les tâches à
réponse fermée, réponses ouvertes enregistrées pour examen humain. Les historiques
des exercices multitours sont fournis par le jeu de données. Aucun ancien tour
n'est supprimé silencieusement pour ces épreuves. Ce protocole ne teste donc pas
une conversation autonome de bout en bout.

Le seuil de régression linguistique est un contrôle statistique limité. Le
checkpoint `best.pt` est le meilleur selon ce critère et cette validation ; il ne
prouve pas une capacité générale supérieure. Aucune promotion en production n'est
automatique. Le rapport marque explicitement cette absence d'éligibilité.

### Résultat du pilote de développement

Le [rapport complet](../research/kira_model/results/dialogue-cpu-pilot.json)
conserve les paramètres, empreintes, historiques et toutes les réponses des sondes.
Sur 30 exercices fermés de validation, la référence initiale réussit **0/30**,
l'adaptation sans répétition **8/30**, et celle avec répétition **8/30**. Les
réponses ouvertes restent souvent incorrectes ou incohérentes. Le diagnostic ne
qualifie donc pas cette version comme assistante généraliste.

| Mesure | Départ | Dialogue sans répétition | Dialogue avec répétition |
|---|---:|---:|---:|
| Perte de dialogue, meilleurs poids | 6,9550 | 1,9259 | 1,8329 |
| Perte littéraire par token BPE | 4,8495 | 6,9803 | 4,9425 |

La perte de dialogue et la perte littéraire portent sur des cibles différentes ;
elles ne se comparent pas entre lignes. La répétition réduit ici l'oubli du corpus
antérieur. Le second essai sélectionne l'étape 1 500 sur 2 000, avant la dernière.
Les 51 tests techniques réussis vérifient la chaîne, pas le niveau d'intelligence.
La vitesse du runner de développement ne mesure pas le PC du propriétaire.

## Commandes

L'environnement PyTorch/Tokenizers CPU déjà installé suffit. Depuis la racine du
code mis à jour, utiliser les poids de son essai BPE et le corpus français vérifié :

```powershell
$kiraLabPython = "$env:USERPROFILE\KIRA-Lab\.venv\Scripts\python.exe"
& $kiraLabPython -m research.kira_model.dialogue prepare --output research/kira_model/data/dialogue-v1/corpus.jsonl
& $kiraLabPython -m research.kira_model.dialogue train --base CHEMIN_VERS_SES_POIDS_BPE --data research/kira_model/data/dialogue-v1/corpus.jsonl --replay-corpus research/kira_model/data/fr-literature-v1/corpus.jsonl --output research/kira_model/runs/dialogue-001 --steps 2000 --stop-after 50
```

Le segment de 50 étapes mesure la vitesse sur le PC. La durée du pilote de
développement ne remplace pas cette mesure. Le chemin du corpus français peut
différer si une préparation de récupération a été utilisée.

```powershell
# Mêmes options mathématiques et mêmes données ; steps reste le total prévu.
& $kiraLabPython -m research.kira_model.dialogue train --data research/kira_model/data/dialogue-v1/corpus.jsonl --replay-corpus research/kira_model/data/fr-literature-v1/corpus.jsonl --output research/kira_model/runs/dialogue-001 --steps 2000 --resume
& $kiraLabPython -m research.kira_model.dialogue evaluate --checkpoint research/kira_model/runs/dialogue-001/best.pt --data research/kira_model/data/dialogue-v1/corpus.jsonl --output research/kira_model/runs/dialogue-001/validation.json
& $kiraLabPython -m research.kira_model.dialogue chat --checkpoint research/kira_model/runs/dialogue-001/best.pt
```

Le nombre total d'étapes fixe le calendrier du taux d'apprentissage ; il ne peut
pas changer pendant une reprise. `--stop-after` arrête un segment sans changer ce
calendrier. Les validations qui sélectionnent les poids suivent le même calendrier
dans un run continu et dans un run repris. La reproduction exacte est testée dans
le même environnement CPU ; elle n'est pas garantie entre matériels et systèmes.
Une reprise vérifie corpus, données, tokenizer, code, options et versions. Pour
une nouvelle phase plus longue, partir explicitement de son `best.pt` dans un
nouveau dossier. Charger seulement ses checkpoints de confiance.

## Direction de développement

Le résultat actuel demande avant tout un corpus français plus diversifié, une base
linguistique plus solide et davantage de calcul. L'apprentissage d'instructions ne
remplace pas ce préentraînement. Les exercices synthétiques ne suffisent pas à
créer un assistant généraliste. Une prochaine version devra enrichir les données
avec provenance et droits vérifiés, puis mesurer la grammaire, la compréhension,
le suivi d'instructions et les régressions sur de vrais cas indépendants.

L'intégration au core viendra après ces évaluations : mémoire explicite consultable,
calcul et recherche avec résultats observés, permissions locales déjà présentes,
modèle attribué à chaque réponse et remplacement réversible. Un moteur tiers
éventuel reste nommé comme tel ; il n'est pas présenté comme l'inférence de nos poids.

Inspirations publiques, sans revendication d'accès aux architectures privées :
[Hugging Face, apprentissage supervisé du dialogue](https://huggingface.co/learn/smol-course/en/unit1/3),
[Anthropic, workflows d'agents](https://www.anthropic.com/research/building-effective-agents),
[Anthropic, évaluations d'agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).
