-- À exécuter dans le SQL Editor Supabase avant l'activation des embeddings.
-- Les tables métier et métadonnées sont migrées de façon additive au démarrage Render.
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;
-- KIRA détecte le schéma de l'extension même si elle était déjà installée ailleurs.
