\set ON_ERROR_STOP on

\if :{?database_name}
\else
    \echo 'database_name is required, for example: -v database_name=ai_learning_rag'
    \quit
\endif

-- Run as a PostgreSQL administrator after creating database ai_learning_rag.
-- Login users receive these NOLOGIN roles; passwords never belong in this file.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ai_learning_rag_migrator') THEN
        CREATE ROLE ai_learning_rag_migrator NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ai_learning_rag_app') THEN
        CREATE ROLE ai_learning_rag_app NOLOGIN;
    END IF;
END
$$;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS rag AUTHORIZATION ai_learning_rag_migrator;

GRANT CONNECT ON DATABASE :"database_name" TO ai_learning_rag_migrator;
GRANT CONNECT ON DATABASE :"database_name" TO ai_learning_rag_app;
GRANT USAGE ON SCHEMA rag TO ai_learning_rag_app;

ALTER DEFAULT PRIVILEGES FOR ROLE ai_learning_rag_migrator IN SCHEMA rag
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO ai_learning_rag_app;
ALTER DEFAULT PRIVILEGES FOR ROLE ai_learning_rag_migrator IN SCHEMA rag
    GRANT USAGE, SELECT ON SEQUENCES TO ai_learning_rag_app;
