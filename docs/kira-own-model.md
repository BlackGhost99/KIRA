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

Exécution depuis la racine du dépôt, dans un environnement séparé (Python 3.10+) :

```bash
python -m venv .venv-model
# Linux/macOS : source .venv-model/bin/activate
# Windows PowerShell : .\.venv-model\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r research/kira_model/requirements.txt
python -m unittest discover -s research/kira_model/tests -v
python -m research.kira_model train --corpus research/kira_model/fixtures/smoke.jsonl --output research/kira_model/runs/seed-001
python -m research.kira_model generate --checkpoint research/kira_model/runs/seed-001/weights.pt --prompt "Une chercheuse" --max-new-tokens 100
```

Le script ne fait aucun appel réseau, ne lit pas la base KIRA et n'utilise aucune clé.
Il refuse d'écraser un dossier de run existant. Il sauvegarde `weights.pt` et
`manifest.json` : configuration, initialisation, seed, version PyTorch, empreinte du
corpus, provenance par document, tokens vus et pertes de validation avant/après.
Le contexte glisse pendant la génération ; sa longueur se mesure en octets UTF-8,
pas en mots. Les checkpoints ne reprennent pas l'état de l'optimiseur : pas encore
de reprise exacte d'un long entraînement. Charger uniquement ses propres checkpoints
locaux de confiance, même avec `weights_only=True`.

Chaque ligne JSONL d'un futur corpus doit porter `id`, `source`, `license`,
`training_allowed: true`, `split` (`train` ou `validation`) et `text`.
L'autorisation renseignée est une déclaration humaine, pas une vérification juridique
automatique. La préparation refuse les doublons normalisés et les identifiants répétés.
Elle ne détecte pas encore les quasi-doublons, les chevauchements de passages, ni les
secrets : réviser et séparer les documents sources avant import. Les conversations et
fichiers privés ne deviennent pas automatiquement des données d'entraînement.

## Étapes suivantes et critères de passage

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
- Calcul : seulement le PC i5/8 Go, autre ordinateur disponible, ou GPU distant ?
- Première priorité proposée : français puis dialogue et tutorat scientifique,
  cohérents avec la vocation d'apprentissage de KIRA ; à ajuster avec le propriétaire.

Ces réponses changent la taille, le corpus et le calendrier ; elles ne bloquent pas
la validation technique de KIRA Seed.
