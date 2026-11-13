\set ON_ERROR_STOP on

\set project_database ai_learning_rag
\set test_database ai_learning_rag_test

-- CREATE DATABASE cannot run inside a DO block. psql's \gexec executes only
-- the missing database statements, so this bootstrap is safe to repeat.
SELECT format('CREATE DATABASE %I', :'project_database')
WHERE NOT EXISTS (
    SELECT 1 FROM pg_database WHERE datname = :'project_database'
) \gexec

SELECT format('CREATE DATABASE %I', :'test_database')
WHERE NOT EXISTS (
    SELECT 1 FROM pg_database WHERE datname = :'test_database'
) \gexec

\connect :project_database
\set database_name :project_database
\ir bootstrap_roles.sql

-- Local development uses the two least-privilege roles as direct login roles.
-- \password prompts securely and does not store credentials in this file.
ALTER ROLE ai_learning_rag_migrator LOGIN;
ALTER ROLE ai_learning_rag_app LOGIN;
\password ai_learning_rag_migrator
\password ai_learning_rag_app

\connect :test_database
\set database_name :test_database
\ir bootstrap_roles.sql

SELECT current_database() AS database_name,
       extversion AS pgvector_version
FROM pg_extension
WHERE extname = 'vector';
