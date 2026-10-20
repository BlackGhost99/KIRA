# KIRA : continuité, projets et travail durable

Ce lot s'appuie sur les PR #1 (mémoire V3) et #2 (sessions et fournisseurs). Il couvre les points 2 à 12 de la feuille de route demandée ; l'essai de production dépend encore de l'accès Render et de la disponibilité du projet Supabase KIRA.

## État de l'essai en ligne

Le 19 octobre 2026, le projet Supabase nommé KIRA était `INACTIVE`. Sa réactivation a été refusée par Supabase parce que le propriétaire possède déjà deux projets gratuits actifs. Aucun autre projet n'a été mis en pause, supprimé ou changé de formule. L'intégration Render disponible n'est pas connectée à cette session. Aucune fusion ni mise en ligne n'a été effectuée.

Une fois ces accès disponibles, déployer le commit validé par CI et vérifier `/api/readiness` après connexion : vraie base PostgreSQL, fournisseurs prêts, révision et statut de l'index sémantique. Tester un échange réel et un souvenir après redémarrage. Les fournisseurs de production et la qualité des embeddings restent à tester avec les clés configurées dans Render, jamais dans le chat.

## Ce qui est ajouté

| Point | Fonctionnement livré | Vérification prévue |
|---|---|---|
| 2. Mémoire vérifiable | Statut confirmé/déduit/à vérifier, liens vers échange et message d'origine, confirmation propriétaire, historique conservé. | Provenance et confirmation versionnées. |
| 3. Tâches persistantes | File en base, progression, résultats, bail renouvelé, annulation coopérative et reprise des tâches idempotentes. | Réouverture de base, concurrence de travailleurs, arrêt avant l'étape suivante. |
| 4. Confidentialité | Un résultat machine privé marque durablement la conversation ; seuls les fournisseurs approuvés sont ensuite éligibles. | Panne du fournisseur approuvé, refus de bascule et refus de fuite au tour suivant. |
| 5. Accord exact | Empreinte des demandes machine et des propositions ; fichiers testés figés, révision de base vérifiée avant ouverture de PR. | Refus d'une empreinte périmée, d'une proposition modifiée ou non testée. |
| 6. Projets | Objectif, contexte, tâches et résultat ; conversation attachée au projet. | Contexte repris et conflit entre deux versions détecté. |
| 7. Documents | PDF, DOCX, TXT, MD ; recherche dans les passages avec page PDF ou section textuelle. OCR français/anglais pour les PDF scannés. | Texte réellement extrait, référence vérifiable, déduplication et OCR d'un PDF image. |
| 8. Apprentissage | Cartes question/réponse/source, échéances de révision, maîtrise déclarée et contexte fourni à KIRA. | Progression durable et refus d'une double révision. |
| 9. Évolution mesurée | Cinq scénarios synthétiques, résultat et consommation par cas, révision identifiée, historique des évaluations. | Résultats sauvegardés, aucun outil réel exécuté par l'évaluation. |
| 10. Appareils | Données partagées côté serveur, versions des projets/tâches/cartes et refus des écritures périmées. | Contrats de fichiers Unicode et de périmètre exécutés sur Linux/Windows/macOS. |
| 11. Voix | Dictée manuelle français/anglais, texte corrigeable et lecture facultative. | Accord explicite, début/arrêt et absence d'envoi automatique dans le test DOM. |
| 12. Sauvegarde | Archive compressée vérifiée, contenu personnel, identité, historique, projets, fichiers, révisions et réglages autorisés ; restauration atomique vers une base vide. | Aller-retour réel, refus d'archive altérée et rollback après contenu invalide. |

## Premier parcours

1. Se connecter puis ouvrir **Plus → Réglages**. Choisir les fournisseurs autorisés à traiter les données privées ; le mot de passe peut être redemandé.
2. Ouvrir **Projets**, créer un projet (privé par défaut), préciser son objectif et son contexte.
3. Ajouter une tâche. L'état « Terminée » exige un résultat ; une modification faite sur un autre appareil entraîne un conflit plutôt qu'un écrasement silencieux.
4. Importer un petit document. Dans **Plus**, suivre le traitement. Revenir au projet et cliquer **Actualiser le projet**, puis **Ouvrir le texte** pour contrôler la source.
5. Cliquer **Discuter de ce projet**. La conversation conserve son projet côté serveur lorsqu'elle est rouverte ailleurs. Un nouveau projet choisi ne remplace pas celui d'une conversation existante.
6. Créer une carte d'apprentissage avec sa réponse et sa source ; révéler la réponse, puis choisir « À revoir » ou « Je maîtrise ».
7. Dans **Mémoire**, retrouver l'échange d'origine, corriger ou confirmer les déductions. L'option d'inclusion des souvenirs privés utilise la recherche par mots, sans embeddings.
8. Dans **Plus**, lancer une évaluation puis rouvrir les résultats. Elle exécute cinq scénarios selon le mode et le budget courants ; chaque scénario peut entraîner plusieurs requêtes en cas de bascule entre fournisseurs.
9. Dans les réglages, télécharger une **Sauvegarde restaurable**. L'export exige une confirmation récente du mot de passe.

## Confidentialité

Les nouveaux souvenirs produits par les outils et la consolidation sont marqués « déduits ». Une saisie propriétaire manuelle est confirmée. Les anciennes données sont « à vérifier » : leur provenance n'est pas inventée. Les corrections gardent les versions antérieures.

Les résultats privés de machine rendent la conversation privée pour les tours suivants. Les nouveaux souvenirs issus de ce contexte sont privés ; ils ne sont pas envoyés au service d'embeddings. Leur contenu et les conversations privées sont exclus du rappel public et de la consolidation quotidienne. Les remarques négatives d'une conversation privée sont aussi exclues du contexte d'évolution. Les anciens souvenirs sans provenance fiable nécessitent une revue du propriétaire : aucune classification rétroactive parfaite n'est prétendue.

Les projets sont privés par défaut. Un propriétaire peut explicitement autoriser un projet non privé ou rendre un souvenir public. Déclasser un souvenir privé exige une confirmation récente du mot de passe. Une conversation déjà devenue privée le reste. Les pages de documents ne sont accessibles que par les routes propriétaires, et les documents sont traités comme des données non fiables lors des demandes d'actions machine.

L'autorisation des fournisseurs est persistante dans `kv.llm_trusted_providers`, initialisée par `LLM_TRUSTED_PROVIDERS` si aucun choix n'a été enregistré. Les nouvelles tables et les tables personnelles existantes ont la RLS activée sur PostgreSQL. L'API serveur doit utiliser le rôle propriétaire des tables ; aucun accès public anonyme n'est ajouté.

La dictée du navigateur dispose d'un consentement séparé : certains navigateurs transmettent l'audio à leur propre service. Le routeur LLM ne contrôle pas ce service. La dictée n'est pas activée par défaut ; la page ne soumet jamais automatiquement les paroles. Elle s'arrête lorsque la page est masquée, à la déconnexion, lors d'un changement de conversation ou d'une confirmation du mot de passe. La lecture vocale est déclenchée par un bouton et peut être arrêtée. La disponibilité et les voix dépendent du navigateur ; un essai réel sur téléphone reste nécessaire.

## Tâches et redémarrages

Le bail d'un travailleur est renouvelé toutes les 20 secondes et expire après 120 secondes. Le serveur vérifie les travaux à reprendre au démarrage puis toutes les 20 secondes. Deux travailleurs ne peuvent pas revendiquer simultanément le même travail vivant. Un traitement déjà lancé observe l'annulation entre ses étapes ; une requête réseau ou une extraction en cours n'est pas interrompue instantanément.

La reprise a une sémantique « au moins une fois ». La veille déduplique ses articles ; l'indexation d'un document remplace ses passages dans une transaction. Une analyse d'évolution interrompue attend une relance explicite : elle n'ouvre jamais de PR automatiquement. Une ancienne tâche à fonction Python temporaire, sans type enregistré, devient « interrompue » et ne prétend pas être reprenable.

Le suivi durable concerne les traitements de fond. Un tour de chat qui subit l'arrêt complet du serveur n'est pas automatiquement rejoué : réexécuter ses outils pourrait doubler des effets. Les messages déjà enregistrés restent consultables. Sur Render gratuit, un traitement en attente reprendra lors d'un réveil du service ; aucun fonctionnement continu 24 h/24 n'est promis.

## Documents et OCR

Limites : 4 Mo par fichier, 100 pages PDF, 1 million de caractères au total. L'extraction se fait dans un sous-processus dont l'environnement ne reçoit pas les secrets applicatifs, avec limites de mémoire/CPU sous POSIX et délai maximal de 60 secondes. Les dépendances OCR sont installées dans l'image Docker (`poppler-utils`, Tesseract français et anglais). Les fichiers ne sont pas envoyés à un service OCR externe.

Un fichier non lisible, chiffré, trop coûteux à traiter ou dépourvu d'OCR disponible échoue avec un état visible et garde son original pour une relance ou un effacement. Le découpage Word/texte indique des **sections**, jamais une pagination Word inventée. Les passages sont retrouvés par mots ; les embeddings multilingues de la mémoire restent une activation séparée. Le texte OCR et les réponses doivent être vérifiés sur les documents importants.

## Accords et évolution

L'interface transmet l'empreinte de la demande qu'elle montre. Une demande dont les arguments ont changé est refusée ; le contrôle précède l'envoi à la machine. La confirmation récente du mot de passe reste requise pour les actions sensibles.

Une proposition d'évolution conserve les fichiers complets issus des tests, son diff, la révision de départ et son empreinte. Seule une proposition dont les tests ont réellement réussi peut être approuvée. Si la branche distante a changé ou si les fichiers reconstruits diffèrent, l'analyse et les tests doivent être relancés. Les propositions d'avant cette migration n'ont pas d'artefact figé : elles doivent être régénérées. Une PR reste à examiner et fusionner par le propriétaire.

Le rapport d'évolution indique la méthode et le résultat des tests. Le banc d'évaluation LLM est synthétique et distinct de cette validation du code ; il ne démontre pas une amélioration générale de l'intelligence. Ses résultats dépendent aussi du modèle sélectionné et de sa disponibilité. Comparer des révisions exige de garder un modèle et des conditions comparables. Les révisions d'apprentissage sont des déclarations personnelles, pas une certification automatique du niveau.

En mode ECONOMY, ou avec Groq gratuit comme seul fournisseur, l'analyse utilise un petit périmètre de fichiers entiers réellement modifiables : instructions et contexte limités à 6 000 octets UTF-8, réponse demandée limitée à 1 024 tokens. Elle ne prétend pas analyser tout le dépôt et peut ne trouver aucune proposition. Les corrections restent bornées. Une analyse complète refusée pour sa taille (HTTP 413) tente une seule reprise compacte ; un nouveau refus arrête l'analyse avec une explication. Ce refus propre à la requête ne met plus le fournisseur en pause pour les conversations suivantes. Les vrais quotas (HTTP 429), pannes et erreurs de configuration conservent leurs protections.

## Sauvegarde et restauration

L'archive personnelle est bornée à 32 Mo décompressés, avec lecture par petits lots pendant l'export. Elle a un format versionné et un SHA-256 par table. Ces sommes détectent la corruption ; elles ne constituent pas une signature prouvant l'identité de l'auteur. Ne restaurer que ses propres archives.

L'archive contient le contenu personnel, les fichiers, l'identité et une liste limitée de réglages. Elle exclut les mots de passe/variables Render, sessions d'accès, cookies, jetons d'appareils et codes d'appairage. Les embeddings peuvent être reconstruits. L'archive contient donc des données personnelles sensibles : la conserver dans un emplacement privé.

```bash
# Utilise DATABASE_URL du serveur ; n'écrase jamais une archive existante.
python -m scripts.personal_backup export /chemin/prive/kira-personal.json.gz

# Crée une nouvelle base SQLite ; refuse un fichier déjà existant.
python -m scripts.personal_backup restore /chemin/prive/kira-personal.json.gz --into-sqlite /nouveau/chemin/kira.db

# Cible PostgreSQL séparée, vide et appartenant au rôle serveur.
# Définir l'URL uniquement dans une variable d'environnement, pas dans la commande.
python -m scripts.personal_backup restore /chemin/prive/kira-personal.json.gz --database-url-env KIRA_RESTORE_DATABASE_URL
```

Les tables sont restaurées dans une seule transaction. Une base déjà peuplée est refusée. Les tâches sont restaurées « interrompues » et les propositions perdent leur validation ; aucune action ne repart seule. Les appareils doivent être appairés à nouveau et le propriétaire se reconnecte avec les secrets de son installation. Réactiver/reconstruire ensuite les embeddings si nécessaire. Cette archive ne remplace pas une sauvegarde administrateur complète de l'instance PostgreSQL et de ses rôles.

## Validation et limites terrain

La CI exécute : suite Python, contrats mémoire/sessions/continuité sur PostgreSQL réel, OCR, parcours DOM sessions/projets/voix, et contrats du cœur sur trois OS. Les tests utilisent des fournisseurs simulés et des bases jetables. Les contrôles DOM ne sont pas une vérification visuelle et ne valident pas un service de dictée réel. Windows/macOS sont validés pour les contrats de fichiers et de périmètre sélectionnés ; le contrôle écran/souris, l'installation automatique, Android/Termux et le matériel doivent encore être essayés sur de vrais appareils.
