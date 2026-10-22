# Nouvelle direction : KIRA et son propre modèle

Décision du propriétaire, 20 octobre 2026 : reprendre la conception globale, développer
les propres poids de KIRA et améliorer l'assistante autour de ce modèle. Le core installé
en arrière-plan reste prioritaire ; le moteur de calcul peut être dans le cloud et
l'interface graphique reste secondaire.

## Ce que signifie « son propre modèle »

L'objectif retenu est un modèle initialisé de zéro et entraîné sur un corpus choisi,
sans présenter un modèle tiers renommé ou un simple prompt comme un modèle KIRA.
On utilise les méthodes publiées et les bibliothèques adaptées, en citant les sources.
Ce n'est pas une nouvelle architecture scientifique revendiquée.

KIRA reste un système complet : ses poids produisent des prédictions ; son core agit
sur les machines ; son service conserve la mémoire, orchestre les outils et applique
les permissions. Une mémoire mise à jour n'est pas un nouvel entraînement des poids.
Ce travail ne démontre ni conscience ni autonomie générale.

## Architecture cible

| Élément | Rôle et priorité |
|---|---|
| Modèle KIRA | Langage, puis dialogue, raisonnement évalué et appels d'outils appris |
| Core installé | Fonctionnement discret à l'ouverture de session, actions sous permissions locales |
| Moteur cloud | Hébergement du modèle validé et orchestration ; machine de calcul à choisir |
| Mémoire explicite | Origine, correction, suppression, projets, faits et préférences du propriétaire |
| Outils | Calcul vérifié, recherche, documents, actions proposées et résultats observés |
| Interface | Contrôle, conversation et lecture du journal, accessible au besoin |

Le serveur Render actuel n'est pas une infrastructure d'entraînement choisie. On garde
le fonctionnement existant pendant le développement ; aucun effacement de données,
remplacement de fournisseur ou redéploiement n'est fait par cette nouvelle base.
Les éventuels appels à des modèles tiers restent identifiés comme tels et ne sont pas
décrits comme l'inférence du modèle KIRA.

## Ce que nous retenons des travaux existants

- Transformer causal : apprendre le prochain token sans voir les tokens futurs.
- Évaluation sur des documents séparés avant toute présentation de capacités.
- Mémoire consultable et calcul/outils externes plutôt que tout enfouir dans les poids.
- Versions de modèle, provenance des données, contrôles avant remplacement et retour possible.

Références primaires : [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
[préentraînement depuis zéro (Hugging Face)](https://huggingface.co/learn/llm-course/en/chapter7/6),
[attention causale PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
Les principes sont réimplémentés dans ce laboratoire ; aucun code ni poids tiers
de modèle n'y est téléchargé. PyTorch reste une dépendance tierce.

## Première étape livrée : KIRA Seed 0.1

`research/kira_model` contient un petit décodeur Transformer, un encodeur UTF-8 par
octets, une chaîne d'entraînement CPU et la génération à partir des poids sauvegardés.
La configuration par défaut est volontairement minuscule : largeur 64, 2 couches,
4 têtes, contexte de 128 tokens/octets, embeddings de positions appris, poids de sortie
liés aux embeddings d'entrée. Pas de tokenizer préentraîné.

Le corpus `fixtures/smoke.jsonl` contient huit paragraphes synthétiques écrits pour
tester la chaîne, dont deux réservés à la validation. **Il ne constitue pas un corpus
pour une assistante utile.** Une baisse de perte ici prouve l'apprentissage statistique
sur ce petit domaine ; elle ne prouve pas une capacité de dialogue ou de raisonnement.
Une sortie incohérente ou des caractères Unicode invalides sont possibles.

Essai CPU effectué : **124 800 paramètres, 200 étapes**, perte de validation de
**5,5482 à 2,5249**, neuf tests réussis. Le test de génération reste répétitif et
incohérent. Le [rapport reproductible](../research/kira_model/results/smoke-cpu.json)
conserve les métriques, les empreintes et cet échec qualitatif : ces chiffres ne sont
pas un score d'intelligence ni une comparaison avec un autre modèle.
Cet essai initial utilisait des lots de quatre séquences et deux threads. Les
nouveaux réglages par défaut sont plus prudents : **un lot de une séquence et un
thread CPU**, sans GPU. Le premier entraînement sur le PC du propriétaire est décrit ci-dessous ; les
chiffres du rapport initial ne mesurent pas son débit.

Exécution depuis la racine du dépôt, dans un environnement séparé (Python 3.10+) :

```bash
python -m venv .venv-model
# Linux/macOS : source .venv-model/bin/activate
# Windows PowerShell : .\.venv-model\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r research/kira_model/requirements.txt
python -m unittest discover -s research/kira_model/tests -v
python -m research.kira_model train --corpus research/kira_model/fixtures/smoke.jsonl --output research/kira_model/runs/seed-001
python -m research.kira_model generate --checkpoint research/kira_model/runs/seed-001/weights.pt --prompt "Une chercheuse" --max-new-tokens 100
# Reprise : mêmes options et corpus ; steps indique le total visé, pas les étapes à ajouter.
python -m research.kira_model train --corpus research/kira_model/fixtures/smoke.jsonl --output research/kira_model/runs/seed-001 --steps 400 --resume
```

Le script ne fait aucun appel réseau, ne lit pas la base KIRA et n'utilise aucune clé.
Il refuse d'écraser un dossier de run existant. Il sauvegarde `weights.pt` et
`manifest.json` : configuration, initialisation, seed, version PyTorch, empreinte du
corpus, provenance par document, tokens vus et pertes de validation avant/après.
Le contexte glisse pendant la génération ; sa longueur se mesure en octets UTF-8,
pas en mots. `training.pt` conserve toutes les 25 étapes (réglable avec
`--checkpoint-every`) les poids, l'optimiseur et les générateurs aléatoires. La
sauvegarde est remplacée atomiquement ; après interruption, `--resume` repart du
dernier checkpoint complet, au prix de perdre les étapes suivantes. Une reprise
refuse tout changement de corpus, architecture, lot, seed, taux d'apprentissage,
nombre de threads ou version PyTorch. Les exports `weights.pt` et `manifest.json`
ne sont actualisés qu'à la fin du run. Ne pas lancer deux entraînements sur le même
dossier. Charger uniquement ses propres checkpoints locaux de confiance, même avec
`weights_only=True`. La reproduction exacte entre OS, versions et matériels n'est
pas garantie ; le test de reprise compare les poids dans le même environnement CPU.

La compatibilité CPU est maintenant confirmée par les sorties du propriétaire :
Python 3.15.0, PyTorch 2.14.1+cpu, runtime Microsoft Visual C++ x64 et NumPy 2.5.4.
Les douze tests initiaux ont réussi sous Windows en 27,734 secondes. Le premier
entraînement local a atteint 200 étapes, avec une perte de validation de
5,5481718448 à 2,5411603551 ; la génération reste incohérente. Il s'agit du PC
i5/8 Go et du petit corpus initial, pas d'une mesure sur le nouveau corpus.

Chaque ligne JSONL d'un futur corpus doit porter `id`, `source`, `license`,
`training_allowed: true`, `split` (`train`, `validation` ou `test`) et `text`.
L'autorisation renseignée est une déclaration humaine, pas une vérification juridique
automatique. La préparation refuse les doublons normalisés et les identifiants répétés.
Elle ne détecte pas encore les quasi-doublons, les chevauchements de passages, ni les
secrets : réviser et séparer les documents sources avant import. Les conversations et
fichiers privés ne deviennent pas automatiquement des données d'entraînement.
La lecture JSONL est limitée à **16 Mio** : le corpus reste en mémoire dans cette
version. Ce n'est pas un lecteur adapté à un entraînement massif.

## Deuxième expérience : français littéraire et mesure CPU

Le module `prepare_fr` prépare un petit corpus reproductible à partir de quatre
fichiers texte de Project Gutenberg. La sélection sépare les ouvrages avant leur
découpage :

| Usage | Ouvrage | Passages retenus |
|---|---|---|
| Entraînement | Jules Verne, *Le tour du monde en quatre-vingts jours*, no 800 | 235 |
| Entraînement | Voltaire, *Candide*, no 4650 | 131 |
| Validation | Maupassant, *Contes de la Bécasse*, no 11714 | 8, répartis sur l'ouvrage |
| Test réservé | Flaubert, *Trois contes*, no 12065 | 8, répartis sur l'ouvrage |

Le corpus JSONL fait 782 827 octets, dont 642 949 tokens/octet côté entraînement
(marqueurs compris). Le script télécharge quatre sources pour environ 1,15 Mo,
chacune limitée à 2 Mio et comparée à une empreinte SHA-256 vérifiée. Une mise à jour
du fichier distant provoque un refus, pas un changement silencieux du corpus.
Les originaux complets et leurs mentions sont conservés dans `raw/`, avec les
attributions et empreintes dans `sources.json`. Les catalogues indiquent le statut
« public domain in the USA » et les fichiers contiennent les conditions Project
Gutenberg ; cela ne constitue pas une vérification automatique des droits dans tous
les pays. Les données téléchargées restent hors du dépôt Git.

Les en-têtes et pieds de page Gutenberg sont exclus des données d'apprentissage ;
des titres et tables des matières peuvent rester dans le corps. Le découpage n'a
pas de chevauchement et les doublons de passages normalisés sont supprimés. Il ne
s'agit pas d'une déduplication de toutes les phrases. Le contenu est historique et
littéraire, avec les biais et le vocabulaire de ses auteurs ; ce n'est ni un corpus
de dialogue, ni une base factuelle moderne, ni une couverture représentative du
français. Le split `test` n'est utilisé ni pour apprendre ni pour choisir la perte
de validation. Son contenu ne doit être évalué qu'après les choix de développement.

Depuis la nouvelle copie du code, avec le même environnement Python déjà installé :

```powershell
$kiraLabPython = "$env:USERPROFILE\KIRA-Lab\.venv\Scripts\python.exe"
& $kiraLabPython -m research.kira_model.prepare_fr --output research/kira_model/data/fr-literature-v1
& $kiraLabPython -m unittest discover -s research/kira_model/tests -v
& $kiraLabPython -m research.kira_model train --corpus research/kira_model/data/fr-literature-v1/corpus.jsonl --output research/kira_model/runs/fr-cpu-001 --width 128 --layers 4 --heads 4 --context 256 --steps 50
```

Ce second profil comporte **859 136 paramètres** et augmente la largeur et le nombre de couches, mais conserve un
thread et un lot de une séquence. Il commence avec de nouveaux poids : architecture
et corpus différents interdisent la reprise du premier essai. Celui-ci est conservé.
Ne pas comparer directement les valeurs de pertes des deux corpus comme une mesure
d'amélioration. Le premier run de 50 étapes sert à mesurer la vitesse et la mémoire
pratique, sans promettre un texte cohérent. Le manifeste et la sortie finale
indiquent `seconds_per_step`, mesuré sur le segment d'entraînement, sauvegardes
comprises mais hors évaluation initiale/finale. Une durée extrapolée reste indicative.

Pour une reprise ultérieure, conserver tous les paramètres du profil et utiliser
`--steps TOTAL --resume`. Les 19 tests vérifient aussi le nettoyage, la séparation
par livre, les empreintes, l'import hors réseau et l'absence d'influence du texte
réservé au test sur les poids appris. Le temps de calcul mesuré dans l'environnement
de développement ne remplace pas la mesure sur la machine du propriétaire.

L'essai local communiqué le 22 octobre 2026 a atteint **1 000 étapes** :
validation initiale **5,6000757116**, finale **2,4263584869**, avec
**0,215174 seconde par étape** sur le segment mesuré. À 50 étapes, le propriétaire
avait mesuré une validation de 2,8587574996. Le texte généré après 1 000 étapes
reste incohérent. Cette baisse confirme un apprentissage sur ce corpus, sans
établir une compétence de conversation.

## Troisième expérience : vocabulaire appris sur le corpus

L'[expérience BPE](kira-seed-bpe.md) ajoute un découpage en sous-mots, entraîné
uniquement sur le split d'apprentissage. Elle réutilise le corpus déjà préparé,
commence avec de nouveaux poids et conserve les checkpoints par octets. La perte
par token change d'unité ; un indicateur en bits par octet UTF-8 est également
enregistré. Aucun gain de qualité du texte n'est supposé sans observer le pilote.

## Étapes suivantes et critères de passage

La [version Dialogue 0.3](kira-seed-dialogue.md) poursuit les poids BPE existants,
ajoute un curriculum original et des évaluations, puis un terminal de dialogue.
La sélection des poids prend en compte une régression éventuelle de la langue.
Cette chaîne reste expérimentale et ne démontre pas une capacité généraliste.

1. **Cadrer une première compétence** : français de base et dialogue simple proposés.
   Fixer les moyens matériels et la limite de dépense avant de dimensionner le modèle.
2. **Constituer le corpus** : sources autorisées, couverture linguistique, qualité,
   déduplication par source, séparation entraînement/validation/test ; test indépendant
   tenu à l'écart des décisions d'entraînement. Aucun téléchargement massif lancé ici.
3. **Choisir la taille** : mesurer débit et mémoire sur la machine choisie, puis estimer
   durée et coût à partir de ces mesures. Le PC i5-2520M / 8 Go peut servir aux petits
   essais ; les capacités finales ne sont pas promises sur ce matériel.
4. **Préentraîner une version utile** : tokenizer plus efficace, calcul GPU, checkpoints
   reprenables, suivi des expériences. Préentraînement depuis zéro conservé.
5. **Apprendre le dialogue** : données d'instructions vérifiées, puis protocoles d'outils.
   Les exemples générés par d'autres IA ne sont utilisés qu'après vérification de leurs
   conditions, de leur qualité et de leur provenance.
6. **Évaluer les capacités** : tests français, calcul avec outil, incertitude, mémoire,
   confidentialité et refus d'actions hors permissions. Mesurer aussi les régressions.
7. **Intégrer à KIRA** : service d'inférence authentifié, attribution de la version,
   essais sans actions réelles, puis activation volontaire du modèle validé.

Le modèle Seed n'est pas connecté aux machines et ne remplace pas Groq dans cette
étape. On ne déploie pas un prototype linguistique comme moteur d'actions personnelles.

## Décisions encore nécessaires

- Le départ reste à **0 € de services payants**, conformément à la préférence exprimée.
  Un entraînement plus ambitieux sera dimensionné avec les ressources disponibles ;
  aucune réservation de GPU payant n'est lancée.
- Calcul confirmé : seulement le PC i5-2520M / 8 Go, sans carte graphique dédiée.
  L'import CPU et l'essai de 1 000 étapes sur le corpus français sont réussis ;
  la qualité du texte reste insuffisante. Une ressource distante gratuite éventuelle n'est ni
  réservée ni supposée disponible.
- Première priorité proposée : français puis dialogue et tutorat scientifique,
  cohérents avec la vocation d'apprentissage de KIRA ; à ajuster avec le propriétaire.

Ces réponses changent la taille, le corpus et le calendrier ; elles ne bloquent pas
la validation technique de KIRA Seed.
