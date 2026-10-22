# KIRA Seed : expérience BPE sur CPU

Le modèle par octets a appris des régularités mais ne produit pas encore de phrases
cohérentes : l'essai du propriétaire atteint 1 000 étapes et une perte de validation
de 2,4263584869, avec 859 136 paramètres. Cette expérience teste un vocabulaire de
sous-mots appris sur le même petit corpus français. Elle ne promet pas un assistant
fonctionnel après un nombre fixé d'étapes.

## Protocole

- Poids initialisés aléatoirement ; aucun modèle ou tokenizer préentraîné téléchargé.
- Byte-level BPE de la bibliothèque Hugging Face Tokenizers, version 0.23.3 épinglée.
- Entraînement du vocabulaire uniquement sur les documents du split `train`.
- Alphabet de 256 octets, vocabulaire cible de 2 048 identifiants au total, longueur
  des fusions bornée à 32, marqueurs BOS/EOS virtuels hors du texte littéral.
- Corpus inchangé, séparation par ouvrage conservée. La validation est encodée avec
  ce vocabulaire ; le test réservé n'est ni entraîné ni évalué.
- Même décodeur causal : largeur 128, quatre couches, quatre têtes, contexte 256,
  un thread CPU, lot de une séquence. La couche de sortie s'agrandit avec le vocabulaire.
  Le nombre réel de paramètres et de tokens est écrit dans le manifeste.
- Premier pilote limité à 1 000 étapes. La vitesse et la qualité seront mesurées ;
  la vitesse du modèle par octets ne prédit pas celle de ce nouveau profil.

Le vocabulaire BPE est appris par comptage de fragments fréquents sur nos données.
Ce n'est pas un modèle de langage externe. La bibliothèque Tokenizers reste une
dépendance tierce, comme PyTorch. La préparation et l'entraînement sont locaux ;
seule l'installation des dépendances utilise le réseau.

Références primaires :
[entraînement depuis un itérateur](https://huggingface.co/docs/tokenizers/v0.23.1/en/training_from_memory),
[paramètres BpeTrainer](https://huggingface.co/docs/tokenizers/v0.23.1/en/api/trainers),
[version épinglée](https://pypi.org/project/tokenizers/0.23.3/).

## Exécution

Depuis la racine du code mis à jour, avec le même environnement Python que le modèle
par octets et en remplaçant le chemin du corpus si nécessaire :

```powershell
$kiraLabPython = "$env:USERPROFILE\KIRA-Lab\.venv\Scripts\python.exe"
& $kiraLabPython -m pip install --no-deps --only-binary=:all: -r research/kira_model/requirements-bpe.txt
& $kiraLabPython -m research.kira_model.bpe_experiment train --corpus research/kira_model/data/fr-literature-v1/corpus.jsonl --output research/kira_model/runs/fr-bpe-001 --steps 1000
& $kiraLabPython -m research.kira_model.bpe_experiment generate --checkpoint research/kira_model/runs/fr-bpe-001/weights.pt --prompt "Une chercheuse" --temperature 0.5 --max-new-tokens 100
```

Le profil local installe uniquement la roue binaire de Tokenizers. Ses dépendances
déclarées pour le Hub ne sont pas utilisées : sur Python 3.15/Windows, elles peuvent
déclencher une compilation de PyYAML et demander les outils C++ de développement.
Les options `--no-deps --only-binary=:all:` évitent ce téléchargement et toute
compilation. Elles concernent seulement ce complément ; PyTorch et ses propres
dépendances restent installés normalement. Les API de téléchargement/push du Hub
ne font pas partie de ce profil et `pip check` peut signaler cette dépendance omise.
Les tests du modèle bloquent explicitement les imports du Hub et de PyYAML pour
vérifier que notre entraînement local n'en dépend pas.

Ne pas relancer `prepare_fr` dans un dossier contenant déjà `corpus.jsonl`.
L'empreinte du corpus français prévu est
`2df9b1bb28bd71081a6181d49063e27e9e92edf5710b704e9b84afad810eee35`.

Le vocabulaire change : les anciens poids par octets ne sont pas compatibles avec
cette expérience. Le nouveau dossier `fr-bpe-001` préserve `fr-cpu-001`.
La génération par octets reste disponible avec son ancienne commande.

Pour interrompre : Ctrl+C, puis reprendre avec les mêmes options et un total
supérieur à l'étape sauvegardée :

```powershell
& $kiraLabPython -m research.kira_model.bpe_experiment train --corpus research/kira_model/data/fr-literature-v1/corpus.jsonl --output research/kira_model/runs/fr-bpe-001 --steps 2000 --resume
```

`training.pt` conserve toutes les 25 étapes l'optimiseur, les RNG et le tokenizer.
`weights.pt` contient également le tokenizer : la génération ne dépend pas du fichier
`tokenizer.json` exporté pour inspection. Les remplacements de checkpoints sont
atomiques ; les exports finaux sont actualisés seulement à la fin du segment.
La reprise refuse des changements de corpus, tokenizer, options mathématiques,
versions des bibliothèques ou code d'entraînement. La reprise exacte est testée
dans le même environnement ; elle n'est pas promise entre systèmes ou processeurs.
L'empreinte du code inclut les fins de ligne : ne pas modifier les sources entre
l'interruption et la reprise. Ne pas lancer deux entraînements dans le même dossier.

## Lecture des résultats

La perte par token BPE **ne se compare pas directement** à la perte par octet de
l'ancien modèle. Le manifeste rapporte aussi :

```text
bits_per_utf8_byte = perte_moyenne_par_token * nombre_de_cibles
                    / nombre_d_octets_utf8 / ln(2)
```

Chaque cible est comptée exactement une fois par la validation déterministe :
fenêtres consécutives sans chevauchement de cibles, contexte réinitialisé à chaque
fenêtre. EOS est une cible, BOS ne l'est pas ; le dénominateur ne contient que les
octets du texte. Ce protocole inclut donc le coût des marqueurs de fin de document.
La taille du contexte en caractères diffère entre modèles : l'indicateur normalise
l'unité, sans isoler à lui seul l'effet du BPE. Les générations et leurs répétitions
doivent également être examinées.

Le manifeste conserve la configuration, les empreintes du code/corpus/tokenizer,
les versions, les tokens vus, les statistiques par split et la validation
avant/après. Le temps par étape mesure le segment d'entraînement, checkpoints
compris, hors entraînement du tokenizer et évaluations initiale/finale.

## Vérification

Le workflow `KIRA Seed` exécute les tests sur Linux/Python 3.12 et Windows/Python 3.15,
puis un pilote Linux sur le corpus français réel. Il conserve manifeste, tokenizer
et poids du pilote en artefact pendant 14 jours. Ce pilote de développement ne
mesure pas la vitesse du PC i5-2520M du propriétaire.

Tests ajoutés : reconstruction du texte Unicode, vocabulaire indépendant des textes
réservés, dimensions et partage des embeddings, interruption/reprise exacte,
refus des reprises incompatibles, checkpoint autonome, refus des anciens poids et
absence d'influence des textes de test sur les poids appris.

La petite taille du corpus et du modèle limite les résultats. Le BPE peut encore
émettre des fragments ou de l'Unicode invalide pendant la génération ; un vocabulaire
plus efficace ne garantit ni grammaire ni compréhension. Le dialogue et les outils
demanderont des données et des évaluations dédiées.
