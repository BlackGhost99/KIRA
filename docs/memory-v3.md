# Mémoire V3 — premier lot cloud

Ce lot conserve le cerveau sur Render et les souvenirs dans Supabase PostgreSQL.
Il n'installe aucun moteur IA sur les appareils. SQLite reste un support de développement et de tests.

## Comportement

- Types existants conservés : profil, fait, connaissance validée, note.
- Nouveaux types : identité, événement, méthode, projet, objectif, expérience, relation.
- Chaque correction conserve une version complète du souvenir, dans la même transaction.
- « Remplacer l'information » crée un nouveau souvenir et relie l'ancien, devenu `superseded`.
  Le système ne décide pas tout seul qu'une contradiction est vraie ou fausse.
- Une archive reste stockée, mais sort du rappel courant. Aucun souvenir n'est effacé parce qu'il vieillit.
- « Effacer » supprime réellement le souvenir, ses versions et son vecteur. Le journal garde uniquement son identifiant.
- Une identité logicielle stable est créée dans `kv.kira_identity`, puis partagée dans le contexte de conversation.
  Les repères d'identité sont des souvenirs versionnés, éditables par le propriétaire. L'outil `remember`
  et la consolidation ne peuvent pas écrire ce type. Ce mécanisme assure une continuité de données ;
  il ne démontre pas une conscience.
- Recherche hybride : FTS français + similarité cosinus pgvector, fusion Reciprocal Rank Fusion.
  Importance, confiance et date servent au départage. Les vecteurs d'un autre modèle ou d'une autre dimension
  ne sont jamais comparés. Les compteurs d'usage ne changent que lors de l'injection dans le contexte de conversation.
- Panne ou absence d'embeddings : mémorisation, corrections et recherche lexicale restent disponibles.
- L'export inclut maintenant l'identité et l'historique. Les vecteurs sont recalculables à partir des souvenirs.

## Activation après validation du lot

1. Exécuter `supabase/migrations/202610070001_memory_vector.sql` dans le SQL Editor Supabase.
   La migration active l'extension `vector`. Les migrations de tables V2 → V3 sont additives et automatiques
   au démarrage du serveur. La RLS est activée sur les souvenirs, versions, vecteurs et valeurs persistantes,
   sans politique de lecture publique : l'API Render authentifiée est leur point d'accès. La connexion Render
   doit utiliser le rôle propriétaire des tables (ou un rôle de serveur avec BYPASSRLS), jamais la clé `anon`.
2. Fusionner la PR uniquement après examen du diff et des résultats CI. Le blueprint existant déploie `main`
   automatiquement sur Render ; une fusion peut donc déclencher la mise en ligne.
3. Choisir le service d'embeddings. Pour du français, le connecteur est préparé pour BGE-M3 multilingue via
   l'endpoint compatible OpenAI de Cloudflare Workers AI. Un compte et un jeton Cloudflare sont nécessaires.
   Dans Render > Environment, renseigner :

   | Variable | Valeur |
   |---|---|
   | `EMBEDDING_URL` | `https://api.cloudflare.com/client/v4/accounts/ACCOUNT_ID/ai/v1/embeddings` |
   | `EMBEDDING_API_KEY` | Jeton API Cloudflare, saisi uniquement dans Render |
   | `EMBEDDING_MODEL` | `@cf/baai/bge-m3` |
   | `EMBEDDING_DIMENSIONS` | `1024` |
   | `EMBEDDING_TIMEOUT` | `5` |
   | `MEMORY_SEMANTIC_THRESHOLD` | `0.65` comme point de départ à évaluer |

   Les textes à indexer et les requêtes sont transmis au fournisseur choisi. Avec BGE-M3 ci-dessus,
   cela inclut Cloudflare. Rien n'est transmis tant que l'URL et le jeton ne sont pas configurés,
   ou que pgvector n'est pas activé. Choisir Workers Free pour rester dans son quota gratuit ; ce code
   n'active aucun abonnement ni moyen de paiement. La gratuité est limitée par les quotas du fournisseur.
4. Dans le Shell Render, exécuter `python -m scripts.backfill_memory --limit 50`. Répéter jusqu'à `indexed: 0`
   avec `remaining_in_batch: 0`. Un lot interrompu est reprenable ; un échec ne supprime pas de souvenir.
   Le traitement quotidien existant indexe aussi jusqu'à 20 anciens souvenirs par passage.
5. Vérifier des paraphrases françaises, par exemple « PC du boulot » et « ordinateur professionnel ».
   Les tests simulés valident le routage des vecteurs, pas la qualité réelle du modèle. Ajuster le seuil sur
   des exemples pertinents et non pertinents. Un remplacement de modèle exige une réindexation.

### Option Supabase sans autre fournisseur

La fonction `supabase/functions/kira-embedding/index.ts` utilise `gte-small` dans le runtime Supabase.
Ce modèle est limité à l'anglais et tronque les textes à 512 tokens : cette option n'est **pas** le moteur
recommandé pour les souvenirs français de KIRA.

Pour des souvenirs anglais : déployer cette fonction avec la configuration fournie, définir un secret dédié
`KIRA_EMBEDDING_SECRET` côté Supabase, puis mettre ce même secret dans `EMBEDDING_API_KEY` côté Render.
Configurer l'URL de la fonction, `EMBEDDING_MODEL=gte-small` et `EMBEDDING_DIMENSIONS=384`.
La vérification JWT est désactivée uniquement pour cette fonction, car elle vérifie elle-même le secret Bearer.
Ne pas réutiliser les clés `anon` ou `service_role`, ni transmettre de clé dans le chat.

## Validation et limites

- Tests de migration V2, révisions atomiques, archives, remplacement, effacement en cascade,
  continuité de l'identité, autorisation API, normalisation/contrat HTTP, pannes et vecteurs périmés.
- Un job CI distinct utilise une vraie base `pgvector/pgvector:pg16`, sans accès à la base de production.
- Aucun service Supabase ni fournisseur réel n'est appelé par les tests unitaires.
- Le classement vectoriel PostgreSQL est exact, sans index HNSW dans ce lot. Ajouter un index dimensionné
  lorsque le modèle et le volume sont stabilisés ; une colonne non dimensionnée permet la réindexation multiversion.
- L'identité et l'historique survivent à un redémarrage tant que `DATABASE_URL` désigne Supabase.
  Le simple identifiant cloud n'assure pas encore la reprise des jobs en cours : ce chantier est distinct.
- Auth HttpOnly, bascule LLM payant → gratuit, jobs persistants, consentement exact des évolutions et
  runtime multiplateforme ne sont pas inclus dans ce premier lot.

## Références techniques vérifiées

- [Recherche hybride Supabase](https://supabase.com/docs/guides/ai/hybrid-search)
- [Extension pgvector](https://supabase.com/docs/guides/database/extensions/pgvector)
- [Limites du modèle Supabase intégré](https://supabase.com/docs/guides/functions/ai-models)
- [BGE-M3 : modèle multilingue, 1024 dimensions](https://huggingface.co/BAAI/bge-m3)
- [Endpoints Cloudflare compatibles OpenAI](https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/)
- [Quota gratuit Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/platform/pricing/)
