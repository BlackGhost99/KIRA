# KIRA : mémoire du projet (pour Claude Code et pour toute IA qui modifie ce dépôt)

## Ce que c'est
KIRA est l'IA **personnelle** de Brice (Gabon) : un **partenaire / assistant / compagnon** de recherche et d'apprentissage
(physique, maths, informatique, langues, droit), façon Jarvis. Ce n'est **pas** un éditeur de code avec IA intégrée
(pas un « Cursor ») : la conversation d'abord, un ton chaleureux et direct, de l'initiative, du code seulement quand
c'est utile ou demandé. Brice se dit moins à l'aise en physique : toujours partir de l'intuition, puis le formalisme.

Hébergement cloud (le PC de Brice est trop faible) : GitHub → Render (Docker) + Supabase (PostgreSQL), pilotable au
téléphone. Aucun fournisseur d'IA n'est indispensable : le routeur bascule automatiquement.

## Principes non négociables (app/principles.py)
1. Souveraineté du propriétaire : Brice décide, KIRA propose.
2. Aucune action sans trace (table `audit`).
3. Évolution uniquement par proposition validée : jamais de fusion automatique.
4. Séparation cognition/exécution.
5. Honnêteté : pas de source ou de chiffre inventé.

Brice veut une IA qui ne lui fait pas la morale et n'écarte pas de sujets par prudence (voir `system_stable` dans
`app/agent.py`). Cela ne supprime ni les garde-fous techniques ci-dessous ni ceux des modèles fournisseurs : on ne
cherche pas à les contourner.

## Le cœur (agent sur les machines de Brice)
KIRA agit sur les machines de Brice (fichiers, commandes, écran, souris/clavier, état) via `core/kira_core.py` (un fichier,
stdlib seule, Python ≥ 3.8, **sortant uniquement** : il interroge `/api/core/*`, aucun port ouvert) et `app/devices.py`.
- Niveaux par machine et par catégorie : `auto` / `ask` / `deny` (défauts : monitor+fs_read libres, le reste sur accord).
  `fs_delete` n'est **jamais** `auto` (imposé dans `set_policy`, `_policy` et à la lecture). Plafonds côté machine
  (pouvoirs actifs, dossiers, liste sensible, pause locale) : le serveur ne peut pas les élargir.
- Ce que Brice approuve doit être montré EN ENTIER : `write_file` ≤ `MAX_SHOWN_WRITE` (20 000) et `run_python` ≤ `MAX_SHOWN_CODE` (6 000)
  caractères, valeurs identiques côté serveur et côté agent (parité testée). Ne remets jamais de troncature dans `describe`.
- Côté agent, écrire/déplacer vers un programme ou un fichier de démarrage (`PROGRAM_EXTENSIONS`, `AUTORUN_*`) exige que
  `exec` soit activé localement (`_refuse_program`) : sinon fs_write contournerait le plafond « exec ».
- Tout passe par `devices.request_action` (politique + audit). Le résumé/détail montré à Brice est calculé par le serveur
  d'après les arguments réels (`describe`), jamais d'après un texte de l'IA.
- Règles d'étanchéité (`ToolContext`) : contenu web lu dans le tour (`tainted`) ⇒ les catégories « libres » sauf `monitor`
  repassent en « sur accord » ; données machine lues (`private_data`: fs_read/exec/screen) ⇒ `web_search`/`fetch_url` refusés.
  Un résultat de machine est toujours étiqueté « donnée, pas instruction ».
- `core/kira_core.py` et `app/devices.py` partagent le même catalogue `KINDS` : `tests/test_core_agent.py` vérifie la parité.
  Tout nouvel type d'action = serveur + agent + `ACTIONS_HELP` + tests.
- L'agent ne se met jamais à jour seul (un serveur compromis ne doit pas pouvoir changer le code qui tourne chez Brice).
- Jetons de machine : `kdev_…` (stockés hachés), en-tête `Authorization: Device …`. Un jeton propriétaire n'ouvre aucune route
  machine et inversement. Les routes machine sont plafonnées en taille (`DEVICE_BODY_LIMIT`, `RESULT_BODY_LIMIT`).
- `/api/files/{id}` est réservé au propriétaire, `Cache-Control: private, no-store` (captures d'écran) ; l'interface les charge
  avec le jeton (`protectedImage` → URL blob).
- Les résultats rendus par les machines sont effacés après 7 jours (`devices.cleanup`), le journal d'actions après 90.

## Fichiers protégés (l'évolution ne peut pas les modifier)
`app/principles.py`, `app/auth.py`, `app/evolution.py`, `app/net.py`, `app/sandbox.py`, `app/budget.py`, `app/devices.py`,
`app/tools/device_tools.py`, `app/main.py`, les tests du cœur (`tests/test_devices.py`, `test_device_tools.py`,
`test_core_agent.py`, `test_core_api.py`), tout `core/` (hors périmètre autorisé) et tout `web/vendor/`. Les modifier = décision
humaine de Brice, jamais une proposition de KIRA. N'y touche pas sans demande explicite.

## Règles de travail
- **Aucun secret dans le code ni dans un message** : clés et mots de passe uniquement en variables d'environnement
  (Render > Environment, ou `.env` local ignoré par git). Ne demande jamais à Brice de coller une clé dans le chat.
- Les modules lisent `config.settings` **au moment de l'appel** (les tests le remplacent) : ne le copie pas à l'import.
- Pile volontairement légère : Starlette + uvicorn + requests + bs4 + sqlite3/psycopg + `unittest`. Pas de FastAPI,
  SQLAlchemy, httpx, pytest. La couche base (`app/db.py`) utilise des `?` traduits en `%s` pour PostgreSQL ; dates en
  ISO-8601 UTC (texte), JSON en texte.
- Format de messages LLM neutre (`app/llm/base.py`) traduit pour chaque fournisseur ; ajouter un fournisseur =
  une classe `Provider` + l'ajouter au routeur.
- Le chat tourne dans un thread à part (`h_chat`) : la réponse est enregistrée même si le téléphone décroche.
- Interface : `web/` en JS pur, sans CDN (CSP stricte `script-src 'self'`). Pas de `alert`/`confirm` ; actions
  destructrices en deux appuis. Texte de l'interface en français, ton direct, vocabulaire simple.
- Les tests ne doivent jamais appeler un vrai fournisseur ni le vrai réseau (FakeSession / ScriptedProvider).

## Commandes
```bash
python -m unittest discover -s tests -t .      # toute la suite (≈ 50 s : l'évolution relance la suite sur une copie)
python -m unittest tests.test_api              # un module
python scripts/dev_demo.py                     # démo sans clé d'API, mot de passe « demo », port 8765 (relie une machine de démo)
uvicorn app.main:app --reload                  # serveur local (lit .env)
```

## Carte rapide
`app/agent.py` (tour de conversation) · `app/llm/` (fournisseurs + routeur + budget) · `app/tools/` (python, web,
mémoire) · `app/memory.py` · `app/veille.py` + `app/consolidate.py` (veille et mémoire nocturne) ·
`app/evolution.py` (proposition → tests sur copie → PR GitHub) · `app/devices.py` + `app/tools/device_tools.py` + `core/` (le cœur) · `app/api.py` (routes) · `app/main.py` (en-têtes de
sécurité, fichiers statiques) · `web/` (PWA) · `legacy/kira_v1/` (ancienne version, pour mémoire).

## Points à surveiller
- Les sources de veille par défaut (`DEFAULT_SOURCES` dans `app/veille.py`) n'ont pas toutes été vérifiées en ligne.
- Chemin PostgreSQL/Supabase testé seulement par traduction de requêtes, pas contre une vraie base.
- Le cœur n'a été testé que sous Linux : démarrages automatiques Windows/macOS/Termux, capture d'écran et souris/clavier
  Windows/Mac, `os.startfile` ne sont pas vérifiés en vrai. Un aller-retour réel serveur ↔ agent est testé (uvicorn en thread).
- Android : pas d'écran ni de toucher (il faudrait une application native) ; seulement fichiers/commandes/état via Termux.
- Offre gratuite Render : mise en veille après 15 min, ≈ 512 Mo de RAM ; la veille quotidienne est réveillée par
  GitHub Actions (`.github/workflows/veille.yml`).
