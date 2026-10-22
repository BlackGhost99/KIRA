# KIRA 0.4 — poursuivre la langue avant le dialogue

Le pilote 0.3 échoue sur les salutations, les variantes d'une question et les tours
suivants d'une conversation. Cette expérience poursuit les **poids BPE propres à
KIRA**, avec le même tokenizer et le même Transformer de 1 088 256 paramètres.
Elle ne charge pas de poids préentraînés d'un autre modèle. Les nouvelles données
externes restent attribuées à leurs auteurs et les textes synthétiques sont signalés.

## Corpus de langue

[LeCarnet](https://huggingface.co/datasets/MaxLSB/LeCarnet) est publié par MaxLSB
et jddqd : histoires françaises courtes à vocabulaire limité, générées avec
Mistral-Large-Instruct-2411, licence déclarée ODC-By-1.0. Nous utilisons un
échantillon, pas les modèles LeCarnet ni l'intégralité des deux millions d'histoires.

`prepare_language.py` épingle la révision
`d0eaae9f8f7f777fde86f16c0b607da1335d2700`, la fiche, un shard de train et le
fichier de validation par taille et SHA-256. Il retient 5 000 histoires de train
et 64 de validation, en conservant les splits officiels. Sélection espacée et
déterministe dans les fichiers, refus des doublons normalisés et des préfixes
identiques de 160 caractères, textes complets et corpus JSONL limité à 14 Mio.
Le corpus observé pèse 6 376 992 octets, SHA-256
`283e4feb5150329b9149dd1b3185a8e227311e906dea39007c14b1ed92fd7536`.
Le fichier de test officiel n'est pas téléchargé.

La lecture Parquet par lots utilise PyArrow uniquement lors de la préparation.
Le JSONL préparé s'entraîne avec l'environnement PyTorch/Tokenizers existant,
sans PyArrow, Hub, datasets ou PyYAML. Le paquet de données joint aux résultats
conserve les fiches, licences, attributions et empreintes.

Ces histoires sont plus simples que les ouvrages littéraires antérieurs. Le
vocabulaire restreint et les constructions répétées limitent aussi leur portée :
ce ne sont ni un corpus de connaissances générales ni des conversations humaines.
Le dédoublonnage exact et par préfixe ne garantit pas l'absence de variantes
sémantiques proches entre les splits.

## Phase linguistique séparée

`language_phase.py` initialise la phase depuis un checkpoint BPE, sans modifier ses
poids ni son vocabulaire au chargement. Un nouvel optimiseur entraîne le prochain
token sur le nouveau corpus. La configuration mesurée est : 10 000 étapes, lot 1,
un thread CPU, contexte 256, pic de LR 0,0005, warmup 100, décroissance cosinus
jusqu'à 10 % du pic. Les gradients sont écrêtés à 1.

L'ancien français est répété sur 25 % des étapes. Le meilleur état doit améliorer
la validation des histoires sans dépasser de 0,35 nat/token la perte initiale de
validation de l'ancien corpus. Le seuil et le calendrier sont fixés avant le run.
Les validations de sélection ont lieu toutes les 500 étapes et à la fin prévue ;
un segment de profilage n'ajoute pas une validation à une autre étape.

La phase conserve état courant, meilleur état, optimiseur, RNG, signatures et
compteurs dans `training.pt` atomique. `weights.pt` exporte l'état sélectionné dans
le format BPE déjà lisible par KIRA ; `last.pt` conserve le dernier. La reprise
est autonome et refuse des données, versions, code ou options mathématiques
différents. Elle garde le nombre total d'étapes prévu. Les anciens modules et
leurs signatures de reprise ne sont pas modifiés.

Depuis la racine du code mis à jour, exemple PowerShell pour mesurer la vitesse
sur le PC du propriétaire ; les chemins doivent désigner les données préparées :

```powershell
$kiraLabPython = "$env:USERPROFILE\KIRA-Lab\.venv\Scripts\python.exe"
& $kiraLabPython -m research.kira_model.language_phase --base CHEMIN_POIDS_BPE --corpus CHEMIN_DONNEES/language/corpus.jsonl --replay-corpus CHEMIN_ANCIEN_CORPUS --output research/kira_model/runs/language-001 --steps 10000 --stop-after 50
& $kiraLabPython -m research.kira_model.language_phase --corpus CHEMIN_DONNEES/language/corpus.jsonl --replay-corpus CHEMIN_ANCIEN_CORPUS --output research/kira_model/runs/language-001 --steps 10000 --resume
```

Le run de développement conserve les poids BPE antérieurs. À 10 000 étapes, la
perte de validation des histoires passe de **5,09086 à 2,32969 nat/token** ; celle
du français antérieur passe de **4,84929 à 4,63249**. Ces validations guident le
développement : ce ne sont pas des tests indépendants de compétence générale.
Le runner mesure 0,06445 seconde par étape, validations et sauvegardes comprises ;
ce chiffre ne mesure pas le PC Windows du propriétaire.

## Dialogue français v2

`prepare_aya_fr.py` épingle
[Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset), révision
`f9ea04583f02a8f86404ff6c58bf75fe637df8a2`, fiche et fichier de train vérifiés par
SHA-256. Il conserve les entrées déclarées françaises, les réponses complètes qui
tiennent dans le contexte et le type d'annotation d'origine, avec la licence
Apache-2.0. Les identifiants des annotateurs ne sont pas importés. Les données
annotées restent susceptibles d'erreurs factuelles ou de langue.

Le corpus final contient **1 109 exemples** : 867 d'Aya, 95 chemins OASST1 déjà
filtrés et 147 exemples synthétiques originaux KIRA. Ces derniers incluent des
paraphrases de présentation, salutations et limites, et douze échanges sur quatre
tours. Les racines normalisées et les entrées ne peuvent pas traverser les splits.
Un conflit de split est écarté plutôt que déplacé dans le train. Les questions
exactes des sondes sont exclues dans tous les tours. La répartition est
**902 train, 111 validation, 96 test**, SHA-256
`0bb849d84500dde26d7a1a812bc13bfb319019ed3b6be5d81f2fda3d2f3dbb0f`.

L'adaptation utilise la chaîne de dialogue existante : rôles virtuels, perte sur
les réponses seules, contexte borné et reprise vérifiée. Tirage équilibré entre
exemples d'interaction et autres dialogues ; les petits exercices arithmétiques
répétitifs de v1 ne constituent plus l'essentiel du train. Un calcul exact reste
disponible par la commande explicite `/calc` du terminal.

Le pilote mesuré part de la phase linguistique sélectionnée : 4 000 étapes, LR
0,0002, warmup 100, répétition des histoires sur 25 % des étapes et contrôle de
régression linguistique de 0,35 nat/token. Le meilleur état est choisi toutes les
200 étapes sur la validation, sans utiliser le test.

Le [rapport linguistique complet](../research/kira_model/results/language-v04-pilot.json)
conserve les données, l'historique et les huit générations avant/après. Les
histoires générées commencent par des phrases plus lisibles mais perdent ensuite
leur cohérence, ignorent certains prompts et inventent encore des mots.

Le [rapport de dialogue](../research/kira_model/results/dialogue-v04-pilot.json)
conserve tous les résultats, y compris les échecs. Après 4 000 étapes, l'export
autonome reproduit ces quatre réponses dans l'ordre, avec son propre historique :

| Question | Réponse générée |
|---|---|
| bjr | Bonjour ! Que souhaites-tu faire ? |
| ca va? | Oui. Je suis disponible pour notre échange. Que souhaites-tu faire ? |
| Comment t'appelles tu ? | Je m'appelle KIRA. |
| T'es sûr que t'es prête ? | Non. Je suis encore expérimentale et je peux me tromper. |

Les autres sessions restent très faibles : présentation hors sujet, demandes
de projet/Python sans réponse pertinente, rappel de la clé remplacé par des
fragments et répétitions. Les exercices fermés anciens régressent de **8/30 à
0/30** sur les mêmes questions de validation. Le train v2 vise le dialogue et
n'inclut plus leurs nombreux templates arithmétiques ; cette régression est
conservée et empêche une promotion généraliste. Le calcul `/calc` reste celui de
l'outil déterministe existant.

Sur les mêmes validations, le pilote 0.3 et le nouveau pilote donnent :

| Perte par token | Pilote 0.3 | Pilote 0.4 |
|---|---:|---:|
| Réponses de dialogue v2 | 5,73962 | 4,40809 |
| Histoires françaises | 5,14535 | 2,40309 |
| Français littéraire antérieur | 4,94251 | 4,97164 |

La dernière ligne est une légère régression, malgré l'amélioration de la phase
linguistique avant adaptation. Les lignes correspondent à des cibles différentes
et ne doivent pas être comparées entre elles. Plusieurs variables changent
ensemble entre les pilotes : corpus, nombre d'étapes et point de départ du
dialogue ; ces chiffres ne sont pas une étude causale d'un facteur isolé.

Le fichier proposé est **un pilote de recherche pour cet essai de base**. Les
treize réponses de l'export final ont été revérifiées identiques à celles du run.
Les 78 tests techniques passent localement avec Hub, PyYAML et PyArrow bloqués ;
ils ne qualifient ni les réponses ni une capacité d'assistante généraliste.

## Vérifier avant de proposer des poids

Les mesures de perte ne suffisent pas. Le rapport doit conserver les générations
et les treize tours de `conversation_eval`, en réinjectant les sorties du modèle.
Les quatre questions du propriétaire sont des régressions connues de développement,
pas un benchmark aveugle. Une proposition comme correction de cet essai nécessite
qu'elles reçoivent toutes une réponse française intelligible et pertinente, avec
identité KIRA et limites expérimentales honnêtes. Les autres sessions et les
anciens exercices doivent aussi être examinés pour signaler les échecs et les
régressions. Un résultat favorable sur ces sondes ne qualifierait toujours pas une
assistante généraliste ou un niveau comparable à ChatGPT/Claude.

Le core installé, le moteur cloud et les permissions d'action restent des éléments
du système à intégrer après qualification. Cette expérience ne change pas leur
déploiement et ne donne pas de contrôle de fichiers au terminal.
