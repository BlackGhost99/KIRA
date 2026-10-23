# KIRA LM 0.5 — entraîner le modèle de KIRA sur Kaggle (GPU gratuits)

Le PC i5-2520M a montré la limite du travail sur CPU : un modèle d'un million de paramètres, trop petit
pour converser. Cette chaîne entraîne **un modèle d'environ 100 millions de paramètres, initialisé de zéro**,
sur les deux GPU T4 gratuits de Kaggle, puis le fait tourner sur le PC. Elle reste dans la règle des 0 €.

**Ce qu'on peut raisonnablement attendre** : un français fluide et un dialogue simple après le préentraînement
et l'adaptation. **Ce qu'il ne faut pas attendre** : des connaissances fiables, du raisonnement solide ou un niveau
ChatGPT/Claude. Les modèles de cette taille inventent des faits ; leurs réponses doivent être relues. Ces attentes
sont des estimations ; seules les sorties réelles, conservées dans les rapports, diront ce qui est atteint.

## Ce qui a changé par rapport au laboratoire CPU

| | KIRA Seed 0.4 (CPU) | KIRA LM 0.5 (Kaggle) |
|---|---|---|
| Paramètres | 1,09 million | ≈ 100 millions (`kira-100m`) |
| Architecture | Transformer GPT simple | Type Llama : RMSNorm, RoPE, SwiGLU, têtes clé/valeur groupées |
| Vocabulaire | BPE 2 048, appris sur nos textes | BPE 32 000, appris sur nos textes (train seulement) |
| Contexte | 256 jetons | 1 024 jetons |
| Données de langue | ≈ 2,5 millions de jetons vus | ≈ 5,4 milliards de jetons préparés |
| Matériel | i5-2520M, 1 cœur | 2 × NVIDIA T4, float16 |

Les poids restent ceux de KIRA : aucun poids d'un autre modèle n'est chargé. L'architecture suit
les conventions publiques de Llama pour pouvoir, plus tard, exporter le modèle vers llama.cpp/Ollama,
un fournisseur que le routeur de KIRA connaît déjà.

## Faits vérifiés sur Kaggle (23 octobre 2026)

- Seul GPU gratuit : **T4 ×2** (2 × 16 Go). Le P100 a été retiré le 15/09/2026.
- Quota : **30 h de GPU par semaine** environ (« parfois plus »). Une heure de T4 ×2 compte pour une heure
  (déclarations du personnel Kaggle de 2022 et 2024, à reconfirmer après la première session).
- Session : **12 h maximum**. Au-delà, l'enregistrement des sorties n'est pas garanti : chaque session
  s'arrête donc d'elle-même vers 11 h.
- `/kaggle/working` : 20 Go conservés à la fin de la session. Écrire beaucoup ailleurs peut faire arrêter la session.
- **Un notebook ne peut pas relire sa propre sortie précédente.** D'où les deux notebooks 2A et 2B qui alternent.
- **Vérification du compte par téléphone obligatoire** pour le GPU et Internet. Personne n'a encore vérifié
  qu'un numéro gabonais est accepté : c'est la première chose à tester.
- Conditions de Kaggle : un seul compte, usage personnel et non commercial.

## Les quatre notebooks

Ils sont dans `research/kira_model/notebooks/` et sont générés par `build_notebooks.py` (un test vérifie
qu'ils sont à jour). Ils récupèrent le code public de ce dépôt au commit utilisé ; ils ne contiennent
ni code recopié, ni clé, ni secret.

| Notebook | Accélérateur | Rôle | Durée estimée |
|---|---|---|---|
| `kira-1-donnees` | Aucun (CPU) | Télécharger, vérifier, apprendre le vocabulaire, écrire les jetons | 2 à 4 h, sans quota GPU |
| `kira-2A-preentrainement` | GPU T4 ×2 | Préentraînement, sessions 1, 3, 5… | ≈ 11 h par session |
| `kira-2B-preentrainement` | GPU T4 ×2 | Préentraînement, sessions 2, 4, 6… | ≈ 11 h par session |
| `kira-3-dialogue` | GPU T4 ×2 | Adaptation au dialogue, 30 sondes, fichier pour le PC | ≈ 1 à 2 h |

Pour en ajouter un sur Kaggle : **Create → New Notebook → File → Import Notebook**, onglet **GitHub**
(`BlackGhost99/KIRA`, branche `main`, chemin `research/kira_model/notebooks/…`) ou onglet **File**.
Dans le panneau de droite, choisir l'accélérateur indiqué, **Internet = On** et
**Environment = Pin to original environment** (PyTorch 2.11 ne doit pas changer en plein run).

## Pas à pas

### 0. Une seule fois

1. Créer le compte Kaggle et le **vérifier par téléphone**.
2. Importer les quatre notebooks.

### 1. Préparer les données

Ouvrir `kira-1-donnees`, ne rien changer, puis **Save Version → Save & Run All (Commit)**. On peut fermer
la page. À la fin, le résumé affiche les jetons par source. La sortie (`donnees/`, ≈ 11 Go) reste attachée
au notebook. Prudence conseillée : dans l'onglet *Output* de cette version, **New Dataset** crée un jeu privé
qui ne disparaîtra pas avec les anciennes versions.

Ne pas relancer ce notebook pendant un préentraînement : de nouvelles données seraient refusées à la reprise
(leur empreinte fait partie de la signature du run).

### 2. Mesurer la vitesse (≈ 20 min de quota)

Dans `kira-2A`, *Add Input* → la sortie de `kira-1-donnees`. Paramètres : `RUN_NAME = "mesure"`,
`NEW_RUN = True`, `MEASURE_STEPS = 30`. Lancer. Le résumé donne les **jetons par seconde** et les milliards de
jetons par heure. Si la mémoire a suffi, essayer `MICRO_BATCH = 16` dans une seconde mesure ; si elle a manqué,
`MICRO_BATCH = 4`. Ce réglage peut changer d'une session à l'autre sans casser la reprise.

### 3. Préentraîner, en alternant A et B

| Session | Notebook | Entrées | Paramètres à changer |
|---|---|---|---|
| 1 | 2A | kira-1-donnees | `RUN_NAME = "kira-100m-a"`, `NEW_RUN = True` |
| 2 | 2B | kira-1-donnees + kira-2A | `RESUME_STEP` = étape affichée à la fin de la session 1 |
| 3 | 2A | kira-1-donnees + kira-2B | `NEW_RUN = False`, `RESUME_STEP` = étape de la session 2 |
| … | … | la sortie de l'autre notebook, **à sa dernière version** | `RESUME_STEP` |

À chaque fois : vérifier dans *Input* que l'entrée de l'autre notebook est à sa dernière version, puis
**Save & Run All**. `RESUME_STEP` est facultatif mais recommandé : si l'entrée n'est pas à jour, le notebook
refuse de partir d'une ancienne étape au lieu de gaspiller une session.

Garde-fous automatiques : pas de nouveau départ sans `NEW_RUN = True` ; pas de `NEW_RUN = True` si le run existe ;
reprise refusée si les données, la forme du modèle ou les réglages d'apprentissage changent ; le code est repris
au même commit que la session précédente (`ALLOW_CODE_CHANGE = True` seulement après avoir lu les changements,
et c'est consigné).

Avec ≈ 30 h par semaine, compter deux sessions de 11 h et une plus courte. Le manifeste de chaque session
garde les pertes, les textes générés (bons ou mauvais) et la vitesse.

### 4. Terminer le préentraînement

Quand tu décides d'arrêter (données vues une ou deux fois, ou progrès qui ralentit) : à la session suivante,
`DECAY_START` = l'étape actuelle et `DECAY_STEPS` ≈ 15 % des étapes déjà faites. Le taux d'apprentissage descend
alors jusqu'à zéro (planning « WSD ») et le run se termine de lui-même. Ce plan ne peut plus changer une fois commencé.

### 5. Adapter au dialogue et rejouer les sondes

Dans `kira-3-dialogue` : entrées = `kira-1-donnees`, `kira-2A` et `kira-2B` (la session la plus avancée est
choisie). Lancer. Le notebook prépare les conversations (Aya et OpenAssistant 2 épinglés + exemples KIRA),
adapte les poids en gardant 20 % de répétition du préentraînement, rejoue les **30 tours de sondes**
(dont « bsr ») et écrit `…-inference.pt` (≈ 200 Mo, float16). Les poids retenus sont ceux de meilleure validation
qui ne dégradent pas le français au-delà du seuil fixé. **Relire les réponses des sondes** : c'est la vraie mesure.

### 6. Discuter avec KIRA sur le PC

Télécharger `…-inference.pt` depuis l'onglet *Output*, puis dans l'environnement du laboratoire :

```powershell
& $kiraModelPython -m research.kira_model.lm_chat chat --weights C:\chemin\vers\kira-100m-a-dialogue-inference.pt
& $kiraModelPython -m research.kira_model.lm_chat probe --weights C:\chemin\...\inference.pt --output sondes-pc.json
```

`/reset`, `/quit`, `/calc expression` fonctionnent comme avant. La vitesse sur l'i5-2520M n'est pas encore mesurée.

## Données

| Source | Révision | Licence | Rôle | Jetons préparés (estimation) |
|---|---|---|---|---|
| [FineWeb2-HQ-Plus](https://huggingface.co/datasets/epfml/FineWeb2-HQ-Plus) `fra_Latn`, 16 fichiers sur 243 | `146690e…` | ODC-By 1.0 + conditions Common Crawl | Web français de meilleure qualité | ≈ 2,5 G |
| [FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu) `fra_Latn`, 4 fichiers sur 42 | `9cfabe2…` | ODC-By 1.0 + conditions Common Crawl | Documents éducatifs | ≈ 1,5 G |
| [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) `frwiki`, 2 fichiers sur 6 | `8bd13e7…` | CC BY-SA 4.0 (texte de Wikipédia) | Encyclopédie | ≈ 1,0 G |
| [LeCarnet](https://huggingface.co/datasets/MaxLSB/LeCarnet), 4 fichiers d'entraînement | `d0eaae9…` | ODC-By 1.0 | Histoires simples, synthétiques (Mistral-Large) | ≈ 0,4 G |

Chaque fichier est épinglé par taille et SHA-256 dans `corpus.py` (registre vérifié le 23/10/2026 via l'API
Hugging Face). Les documents sont nettoyés (NFC, textes trop courts ou abîmés écartés), dédoublonnés toutes sources
confondues, puis répartis par empreinte : 0,1 % en validation, 0,1 % en test, jamais déplacés d'un split à l'autre.
Le test n'est ni appris ni utilisé pour choisir des poids. Le mélange par défaut est 50 % / 22 % / 18 % / 10 %.

Mesure faite sur un échantillon réel : le vocabulaire de 32 000 jetons donne 4,4 à 4,6 octets par jeton sur
ces textes (16 384 jetons : ≈ 4,0). Obligations : citer les sources ci-dessus ; le texte de Wikipédia est sous
CC BY-SA 4.0, ce qui compte si des poids ou des textes dérivés sont un jour publiés.

Conversations (`sft_data.py`) : Aya Dataset (Apache-2.0, annotations humaines), OpenAssistant 2 (Apache-2.0,
conversations humaines relues) et exemples rédigés pour KIRA. Ces derniers ont été écrits avec l'assistance de
Claude Code ; c'est indiqué dans leur origine. Essai réel : 1 715 conversations (1 370 d'entraînement). C'est peu ;
les grands jeux de dialogue traduits automatiquement ou générés par d'autres IA ne sont **pas** ajoutés tant que
leurs conditions n'ont pas été examinées et acceptées. Les questions des 30 sondes, et leurs variantes qui ne
diffèrent que par la ponctuation, sont exclues de toutes les données.

## Ce qui est vérifié, et ce qui ne l'est pas

Vérifié dans l'environnement de développement (CPU, PyTorch 2.14.1 et 2.11.0, la version de Kaggle) :
- 117 tests du laboratoire, dont 39 nouveaux : causalité, cache de génération, RoPE, tokenizer sans rôle injectable,
  préparation déterministe, splits, budgets, reprise **identique au bit près**, refus des reprises incompatibles,
  arrêt sur limite de temps, sélection sous garde de langue, deux processus = un processus (écart nul).
- Simulation complète des quatre notebooks avec de faux dossiers Kaggle : données, sessions 2A → 2B → 2A,
  décroissance finale, dialogue, sondes et export ; refus attendus déclenchés.
- Préparation sur de vrais fichiers épinglés (FineWeb2-HQ-Plus et LeCarnet) : empreintes, Parquet, tokenizer.

Pas encore vérifié :
- Le calcul sur GPU (float16, GradScaler, NCCL entre les deux T4) n'a pas pu être exécuté ici.
- La vitesse réelle, donc le nombre de jetons par semaine : à mesurer à l'étape 2.
- Que Kaggle attache bien la dernière version de l'autre notebook à chaque lancement (à contrôler avec `RESUME_STEP`).
- L'acceptation d'un numéro de téléphone gabonais.
- La qualité du modèle : rien ne la garantit avant d'avoir lu ses vraies réponses.

## Commandes du laboratoire

```bash
python -m research.kira_model.corpus list                     # sources épinglées et licences
python -m research.kira_model.corpus prepare --output D --raw-dir B --threads 4
python -m research.kira_model.gpu_train pretrain --data D --output RUN --preset kira-100m --time-limit 39600
python -m research.kira_model.gpu_train sft --base RUN/weights.pt --conversations C.jsonl --replay-data D --output RUN2
python -m research.kira_model.sft_data prepare --tokenizer D/tokenizer.json --context 1024 --raw-dir B --output C.jsonl
python -m research.kira_model.lm_chat chat|probe|export --weights W.pt ...
python -m research.kira_model.build_notebooks [--check]
```
