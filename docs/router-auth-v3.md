# Routeur cloud et sessions V3

Ce lot s'appuie sur la mémoire V3 de la PR #1. Il ajoute Gemini et OpenRouter aux fournisseurs existants et remplace le jeton navigateur stocké dans localStorage par des sessions côté serveur.

## Bascule et modes

- `QUALITY` suit `LLM_PRIORITY`. Après un échec payant, les candidats gratuits sont essayés avant les autres candidats payants.
- `ECONOMY` privilégie les fournisseurs gratuits et les modèles hébergés par le propriétaire. Le payant est exclu, sauf activation explicite de `LLM_ECONOMY_ALLOW_PAID=true`.
- `PRIVATE`, affiché « Fournisseurs autorisés », limite **les appels au routeur LLM** à `LLM_TRUSTED_PROVIDERS`. Cette liste est vide par défaut : il faut y inscrire les fournisseurs cloud approuvés. Aucun modèle local n'est exigé. Ce réglage ne rend pas Render/Supabase locaux et ne change pas les services de recherche ou d'embeddings.

Le mode choisi dans Réglages est enregistré dans la base et partagé entre appareils. La variable `LLM_ROUTING_MODE` fournit uniquement le choix initial.

Un modèle incapable de lire des images n'est pas appelé quand les messages contiennent une capture. Le routeur vérifie aussi la capacité déclarée d'appeler des outils. Ces déclarations correspondent aux modèles proposés par défaut ; si le propriétaire change un modèle, sa compatibilité doit être vérifiée.

Le budget quotidien porte sur les **jetons payants**. Les fournisseurs gratuits peuvent répondre après son épuisement. Les anciennes lignes de consommation sont classées payantes lors de la migration. Ce compteur de jetons n'est pas un plafond monétaire : un appel peut dépasser le solde restant et plusieurs appels simultanés peuvent se chevaucher.

## Configurer des secours gratuits dans Render

Les clés restent exclusivement dans Render > Environment.

```dotenv
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FREE_TIER=true
GROQ_API_KEY=...
GROQ_FREE_TIER=true
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openrouter/free
LLM_PRIORITY=anthropic,openai,deepseek,gemini,groq,openrouter,ollama
LLM_ROUTING_MODE=QUALITY
LLM_ECONOMY_ALLOW_PAID=false
```

`GEMINI_FREE_TIER` et `GROQ_FREE_TIER` sont des **déclarations du propriétaire**, pas une détection du compte. Elles sont fausses par défaut. N'activer `true` qu'avec un compte/projet réellement gratuit et un modèle éligible. KIRA ne peut pas désactiver la facturation chez ces fournisseurs. Les conditions de traitement des données et les quotas gratuits restent ceux de chaque fournisseur.

OpenRouter accepte uniquement `openrouter/free` ou un identifiant terminé par `:free`. Chaque requête impose un prix maximal de zéro pour les jetons, requêtes et images, et exige la prise en charge des paramètres demandés. Cela n'enlève pas les limites de requêtes du compte gratuit.

## Erreurs et visibilité

Les fournisseurs en pause sont exclus des appels suivants jusqu'à la fin du délai. Une réponse `Retry-After` fixe ce délai ; à défaut, une panne réseau ou HTTP impose 120 secondes, un problème de crédit 600 secondes. Les erreurs de clé, d'accès ou de modèle (401/403/404) désactivent le fournisseur jusqu'à un changement de sa configuration.

`LLM_TIMEOUT=45` fixe le délai de lecture d'une requête (5 à 120 secondes), avec 10 secondes de délai de connexion. Plusieurs fournisseurs en panne peuvent donc allonger la réponse. Les redirections HTTP ne sont pas suivies.

L'API d'état expose classe de coût, compatibilité image/outils, pause, cause publique, dernier délai et compteurs de succès/échecs. Les compteurs sont en mémoire et repartent à zéro au redémarrage ; les consommations et traces de bascule restent en base. Aucun quota restant fournisseur n'est inventé. Les corps d'erreur et clés ne sont pas journalisés.

Les signatures de raisonnement renvoyées par Gemini sont conservées entre les appels d'outils et renvoyées uniquement à Gemini.

## Sessions navigateur

- Cookie d'accès opaque : 15 minutes (`ACCESS_MINUTES`). Cookie de renouvellement opaque : durée fixe de 30 jours (`SESSION_DAYS`), sans prolongation à chaque rotation.
- Cookies `HttpOnly`, `Secure`, `SameSite=Strict`, sans domaine partagé ; chemins `/api` et `/api/auth`.
- La base conserve uniquement des empreintes HMAC des jetons. Les tables de sessions ont la RLS activée sur PostgreSQL pour refuser leur lecture par les rôles publics Supabase.
- Chaque renouvellement remplace les deux jetons. Réutiliser un ancien jeton de renouvellement révoque la session, y compris ses nouveaux jetons.
- L'interface partage un seul renouvellement entre ses requêtes et utilise Web Locks pour le coordonner entre onglets lorsque le navigateur le permet. Sur un navigateur sans Web Locks, une rotation concurrente peut obliger à se reconnecter. Une réponse de renouvellement perdue peut aussi imposer une reconnexion.
- La déconnexion révoque côté serveur puis efface les cookies. Réglages affiche les sessions actives et permet leur révocation.
- Changer `OWNER_PASSWORD` ou `SECRET_KEY` invalide les sessions existantes.

Les mutations par cookies exigent `X-KIRA-CSRF: 1` et refusent une origine différente ou `Sec-Fetch-Site: cross-site`. Définir `PUBLIC_ORIGIN` sur l'URL HTTPS publique exacte, sans chemin ; à défaut `RENDER_EXTERNAL_URL` est utilisé, puis l'origine de la requête. Aucun CORS permissif n'est ajouté. Une authentification bearer d'accès valide peut toujours servir aux clients programmatiques ; les routes machine gardent leur jeton `Device` distinct.

Après cinq minutes (`REAUTH_SECONDS`), le mot de passe est redemandé avant d'ouvrir une PR d'évolution approuvée, appairer une machine, modifier ses politiques, reprendre des machines, approuver une action machine ou révoquer une autre session. Le contrôle précède les effets. Le mot de passe se saisit dans une boîte dédiée et ne part pas dans le chat.

## Migration et développement

À la mise à jour, le navigateur efface l'ancien `kira_token` de localStorage et demande une nouvelle connexion. `/api/auth/login` renvoie le nom et les cookies, **plus aucun jeton JSON**. Les anciens HMAC bearer sont désactivés par défaut. `ALLOW_LEGACY_BEARER=true` permet une transition temporaire pour un client ancien ; cette compatibilité ne permet pas de confirmer les actions sensibles.

Sur Render HTTPS, conserver `COOKIE_SECURE=true`. Pour un serveur de développement HTTP uniquement, définir explicitement `COOKIE_SECURE=false`. La démo locale le fait déjà. Les migrations de tables et de consommation sont additives et idempotentes ; elles ne déploient pas automatiquement le code.

## Vérification

La suite Python utilise des fournisseurs et un réseau simulés. Les tests de sessions vérifient les cookies, CSRF, expiration, rotation concurrente, rejeu, révocation, changement de secrets et confirmation sensible sur SQLite. Ils sont aussi exécutés sur un PostgreSQL réel jetable en CI, avec un contrôle RLS et une migration de consommation V2.

Le parcours DOM de l'interface vérifie restauration de session, renouvellement partagé, changement de mode, mauvais puis bon mot de passe, révocation et déconnexion. Il utilise jsdom et ne constitue pas un contrôle visuel dans un navigateur réel.

```bash
python -m unittest tests.test_llm tests.test_router_v3 tests.test_auth_sessions -v
# CI PostgreSQL : KIRA_TEST_POSTGRES_URL doit désigner une base de test jetable.
python -m unittest tests.test_memory_postgres tests.test_auth_postgres -v
npm install --prefix /tmp/kira-ui-tests jsdom@30.1.2
NODE_PATH=/tmp/kira-ui-tests/node_modules node tests/test_web_sessions.cjs
```

Les véritables clés, quotas, modèles et services Render/Supabase ne sont pas appelés par ces tests. Après validation et déploiement, il reste à vérifier la connexion et un appel par fournisseur avec la configuration réelle.
