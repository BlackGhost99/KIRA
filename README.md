# KIRA

Ton partenaire de recherche et d'apprentissage : une IA personnelle qui te connaît, t'explique la physique, les maths,
l'informatique, les langues ou le droit au niveau où tu en es, calcule et trace pour de vrai, surveille Internet pour toi
et te propose elle-même des améliorations. Tu la gardes sous contrôle : **rien ne s'applique sans ton accord.**

KIRA tourne dans le cloud (ton PC n'a rien à calculer) et s'utilise depuis le téléphone comme une application installée.

## Ce qu'elle sait faire

| | |
|---|---|
| **Discuter** | Conversation avec mémoire. Réponses en Markdown avec formules (LaTeX), tableaux, graphiques. Bouton « Réflexion approfondie » pour les raisonnements difficiles. 👍 / 👎 pour la corriger. |
| **Calculer** | Vrai Python isolé (numpy, scipy, sympy, matplotlib) au lieu de calculer de tête ; graphiques insérés dans la réponse. |
| **Chercher** | Recherche web et lecture de pages, avec les adresses citées. |
| **Se souvenir** | Ce qu'elle sait de toi (niveau, lacunes, objectifs) est lisible, corrigeable et effaçable dans l'onglet *Mémoire*. |
| **Apprendre seule** | Chaque jour, elle lit des sources (arXiv, Quanta, Phys.org…), résume, et te soumet les nouveautés. Tu gardes ou tu écartes : ce que tu gardes entre dans sa mémoire. |
| **S'améliorer** | Elle relit son propre code, ce que tu lui as dit et tes 👎, puis prépare une modification vérifiée par ses tests. Si tu acceptes, elle ouvre une *pull request* sur GitHub : **elle ne fusionne jamais**. |
| **Agir sur tes machines** | Son « cœur » : un petit programme sur ton PC, ton Linux/Mac ou ton téléphone Android lui permet de lire et écrire des fichiers, lancer des commandes, voir l'écran, piloter souris et clavier et surveiller l'état de la machine. **Tu règles chaque pouvoir, machine par machine : libre, sur accord, interdit.** |
| **Ne jamais te lâcher** | Si un fournisseur d'IA tombe en panne (Anthropic, OpenAI, DeepSeek, Groq, ou un modèle local), elle bascule automatiquement sur le suivant. |

## Mise en ligne, pas à pas (depuis un téléphone, ~20 minutes)

### 1. La mémoire durable : Supabase
Un hébergement gratuit efface ses fichiers à chaque redémarrage. Pour que KIRA n'oublie rien :
1. Crée un projet sur [supabase.com](https://supabase.com) (région la plus proche).
2. *Connect* → *Session pooler* → copie l'adresse `postgresql://…`. Remplace `[YOUR-PASSWORD]` par le mot de passe du projet.
   Prends bien le **Session pooler** (et non « Direct connection ») : Render gratuit ne parle qu'en IPv4.
3. Garde cette adresse pour l'étape 2 (variable `DATABASE_URL`).

### 2. Le serveur : Render
1. [render.com](https://render.com) → *New +* → *Blueprint* → choisis le dépôt `BlackGhost99/KIRA`. Render lit `render.yaml`.
2. Renseigne les variables demandées (*Environment*) :
   - `OWNER_PASSWORD` : ton mot de passe de connexion à KIRA (long, unique).
   - `DATABASE_URL` : l'adresse Supabase de l'étape 1.
   - `ANTHROPIC_API_KEY` : clé sur [console.anthropic.com](https://console.anthropic.com). **Fixe-y une limite de dépense mensuelle.**
   - Facultatif : `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY` (secours), `TAVILY_API_KEY` (meilleure recherche web).
   - `SECRET_KEY` et `CRON_TOKEN` sont générés automatiquement.
3. Au premier déploiement, ouvre l'adresse `https://….onrender.com`, connecte-toi, et regarde *Plus → Réglages* : les fournisseurs
   configurés ont un voyant vert.

> **Ne colle jamais une clé d'API dans une conversation, un message ou un fichier du dépôt.** Elles se saisissent uniquement
> dans Render (*Environment*). Si une clé fuit, révoque-la chez le fournisseur et recrée-en une.

Offre gratuite : le service s'endort après 15 minutes sans visite et met jusqu'à une minute à se réveiller (l'écran
l'indique). Il dispose d'environ 512 Mo de mémoire. Le palier payant le plus bas supprime l'endormissement.

### 3. L'évolution et la veille automatique : GitHub
1. **Jeton pour l'évolution** : GitHub → *Settings* → *Developer settings* → *Fine-grained tokens* → limité au seul dépôt
   `KIRA`, droits *Contents: Read and write* et *Pull requests: Read and write*. Colle-le dans Render sous `GITHUB_TOKEN`.
2. **Veille chaque matin** : dépôt → *Settings* → *Secrets and variables* → *Actions* → ajoute
   `KIRA_URL` (l'adresse Render, sans `/` final) et `CRON_TOKEN` (valeur visible dans Render → *Environment*).
   Le fichier `.github/workflows/veille.yml` réveille alors KIRA à 06 h (Libreville). Tu peux aussi le lancer à la main
   (*Actions* → *Veille quotidienne* → *Run workflow*).

### 4. L'installer sur le téléphone
Ouvre l'adresse dans Chrome → menu ⋮ → *Installer l'application* (ou *Ajouter à l'écran d'accueil*).

## Le cœur : KIRA sur tes machines

Le cœur (`core/kira_core.py`) est un seul fichier Python, sans aucune dépendance. Une fois installé, il tourne discrètement
sur la machine (démarrage automatique) : **on ne le voit pas, mais il est là**. C'est lui qui exécute ce que KIRA te
propose et que tu as autorisé. Il appelle le serveur (jamais l'inverse : aucun port ouvert chez toi).

### Relier une machine (2 minutes)
1. Dans l'application : *Plus → Mes machines → Ajouter une machine*. Un code à usage unique apparaît (valable 10 minutes),
   avec les commandes à coller sur la machine, adaptées à Windows, Linux/Mac ou Android (Termux).
2. Colle-les sur la machine. La dernière (`install`) lance le démarrage automatique.
3. La machine apparaît dans l'application. Au début, **tout ce qui modifie quelque chose demande ton accord.**

Python 3 est nécessaire sur la machine (python.org sous Windows, `pkg install python` sous Termux). Facultatif : `pip install
mss pillow` (capture d'écran fiable), `pip install pyautogui` (souris et clavier ; ou `xdotool` sous Linux), `psutil`.

### Tes trois niveaux, par machine et par pouvoir
| Pouvoir | Par défaut |
|---|---|
| État de la machine · Lire des fichiers | **Libre** |
| Écrire · Supprimer · Lancer des commandes · Voir l'écran · Souris et clavier | **Je valide** |

*Libre* : KIRA le fait seule. *Je valide* : une carte s'affiche dans l'application avec **la commande ou le contenu exact**
(calculé par le serveur d'après les vrais arguments, jamais d'après ce que dit l'IA) et deux boutons. *Interdit* : jamais.
Ce que tu approuves est montré **en entier** : une écriture dépasse rarement 20 000 caractères et un script Python 6 000, sinon
KIRA doit découper (écritures « ajouter » successives, chacune visible), pour qu'aucune suite cachée ne suive un début anodin.
Supprimer ne peut jamais être « Libre » ; un fichier supprimé va dans une corbeille récupérable 30 jours
(`python kira_core.py trash list`). Un fichier remplacé est d'abord copié dans cette corbeille.

### Ce que la machine décide elle-même (le serveur ne peut pas l'élargir)
- **Quels pouvoirs sont actifs.** Par défaut : état, lecture, écriture et suppression, limités aux dossiers autorisés.
  Commandes, écran et souris/clavier restent **coupés** tant que tu ne les actives pas sur la machine
  (`python kira_core.py allow exec`, ou `--full` à l'appairage).
- **Pas de programme sans « commandes ».** Tant que « Lancer des commandes » est coupé sur la machine, elle refuse aussi
  d'*écrire* ou de *déplacer vers* un programme ou un fichier qui se lance tout seul (`.sh`, `.bat`, `.exe`, `.ps1`, `.desktop`,
  `.bashrc`, hooks `.git/hooks`, dossiers de démarrage…) : sinon déposer un script contournerait le plafond. Un fichier `.py`,
  `.md` ou `.txt` reste écrivable ; si tu le lances toi-même ensuite, c'est ta décision.
- **Les dossiers autorisés** (par défaut `~/KIRA`) : `python kira_core.py roots add DOSSIER`. Hors de ces dossiers, rien,
  liens symboliques compris. Les clés SSH, `.env`, mots de passe, jetons et le dossier du cœur sont **toujours interdits**,
  même dans un dossier autorisé.
- **La pause** : `python kira_core.py pause` (ou la souris dans un coin de l'écran quand elle est pilotée) coupe tout, tout de
  suite, sans passer par le serveur.
Dans l'application, *Tout mettre en pause* coupe toutes les machines d'un geste, et *Retirer* révoque une machine
immédiatement (son jeton ne vaut plus rien). Chaque demande, accord, refus et résultat est dans le journal.

### Protection contre les pièges
- Ce qui vient d'une machine (fichier, sortie, capture) est traité comme **une donnée**, jamais comme un ordre.
- Après avoir **lu une machine**, KIRA n'a plus le droit de chercher sur le web pendant ce tour : tes données ne partent pas.
- Après avoir **lu Internet** (ou une connaissance issue de la veille), un pouvoir « Libre » repasse en « Je valide » pour
  ce tour : une page piégée ne peut pas la pousser à agir seule (seul « État de la machine » reste libre).
- Les contenus renvoyés par les machines (texte lu, sorties, captures) sont effacés du serveur après 7 jours ; le journal
  (qui a demandé quoi) reste 90 jours.

### Limites à connaître
- Le cœur **ne se met pas à jour tout seul** (volontairement : un serveur compromis ne doit pas pouvoir changer le programme
  qui tourne chez toi). Pour mettre à jour : retélécharge `kira_core.py` et relance `install`.
- Si quelqu'un prenait le contrôle du *serveur*, il ne pourrait agir que **dans les plafonds fixés sur chaque machine** :
  c'est pourquoi les pouvoirs sensibles sont coupés par défaut côté machine. Active-les seulement sur les machines de confiance.
- Capture d'écran et souris/clavier : un seul écran (le principal). **Android** : fichiers, commandes et état via Termux ;
  voir l'écran et toucher le téléphone demanderait une vraie application Android (non incluse).
- Tant qu'une machine est reliée, le serveur reste éveillé (elle l'interroge toutes les 20 secondes) : sur l'offre gratuite de
  Render, cela consomme les 750 h mensuelles d'un seul service, ce qui suffit.

## Comment elle évolue (et pourquoi tu restes aux commandes)

1. Tu lances *Plus → Comment je peux m'améliorer* (avec une piste si tu en as une).
2. KIRA lit son code, ses statistiques d'usage et tes 👎, puis propose une modification **et lance toute la suite de tests
   dessus**. Tu vois l'idée en français ; le détail technique est derrière « Détails techniques ».
3. Si tu acceptes, elle ouvre une *pull request* sur GitHub. Les tests tournent encore là-bas.
4. **C'est toi qui fusionnes** (ou pas). Render redéploie alors la nouvelle version.

Elle ne peut jamais toucher aux fichiers qui font ses garde-fous : `app/principles.py`, `auth.py`, `evolution.py`, `net.py`,
`sandbox.py`, `budget.py`, `devices.py`, `tools/device_tools.py`, `main.py`, aux tests du cœur, à `core/` (le programme des
machines), ni aux bibliothèques embarquées.

## Travailler en local

```bash
pip install -r requirements.txt
cp .env.example .env          # puis remplis OWNER_PASSWORD, SECRET_KEY, une clé d'API…
uvicorn app.main:app --reload # http://127.0.0.1:8000

python scripts/dev_demo.py    # version de démonstration SANS clé d'API (mot de passe : demo)
python -m unittest discover -s tests -t .   # suite de tests
```

## Carte du projet

```
app/            le cerveau (Python, Starlette)
  agent.py        un tour de conversation : mémoire + outils + bascule de fournisseur
  llm/            Anthropic, OpenAI, DeepSeek, Groq, Ollama et le routeur (priorité, pause, budget)
  tools/          calcul Python, recherche web, lecture de pages, mémoire
  veille.py       lecture des sources, résumés, validation
  evolution.py    propositions de modification + pull requests   (protégé)
  sandbox.py      exécution isolée du code de calcul               (protégé)
  devices.py      machines, niveaux, accords, file d'actions       (protégé)
  principles.py   principes fondateurs et périmètre de l'évolution (protégé)
core/           kira_core.py : le cœur installé sur tes machines (un fichier, sans dépendance)
web/            l'application mobile (HTML/CSS/JS sans CDN, installable)
tests/          plus de 300 tests automatiques (dont un vrai aller-retour serveur ↔ machine)
legacy/         l'ancien KIRA v1, conservé pour mémoire
```

## Sécurité en bref

- Un seul propriétaire : mot de passe → jeton signé, délai d'attente après les mauvais essais, journal de toutes les actions.
- Le code de calcul tourne dans un processus séparé, sans accès aux clés du serveur, avec limites de temps et de mémoire.
- Le contenu venu d'Internet est traité comme un document, jamais comme un ordre.
- Machines : jeton propre à chaque machine (stocké haché), appairage par code à usage unique, plafonds fixés sur la machine,
  fichiers sensibles toujours interdits, accord affiché avec la commande exacte, pause et révocation immédiates (voir
  « Le cœur »).
- Les graphiques et captures d'écran ne sont jamais publics ni mis en cache : ils ne se chargent qu'avec ta session.
- Plafond quotidien de jetons (`DAILY_TOKEN_BUDGET`) pour que la facture reste maîtrisée.
- *Plus → Réglages → Exporter mes données* : tout ce que KIRA sait, en un fichier.

## Limites actuelles

- Les modèles d'IA (Anthropic, OpenAI, DeepSeek…) gardent leurs propres garde-fous : KIRA ne les contourne pas.
- Les adresses des sources de veille par défaut sont des flux publics usuels ; certaines peuvent changer. *Veille → Sources*
  permet d'en ajouter, de couper ou de corriger.
- Le cœur n'a été testé ici que sous Linux (y compris un vrai aller-retour avec le serveur). Les démarrages automatiques
  Windows (Planificateur de tâches), macOS (launchd) et Android (Termux:Boot), la capture d'écran et le pilotage souris/clavier
  sous Windows/Mac demandent un premier essai chez toi : dis-moi ce qui coince.
- Les modèles DeepSeek, Groq et Ollama ne savent pas lire les images : si KIRA passe sur l'un d'eux, elle peut agir sur ta
  machine mais ne verra pas une capture d'écran (elle te le dira).
- Un modèle local de secours (Ollama) n'est utile que si tu l'héberges toi-même ailleurs (`OLLAMA_URL`).
