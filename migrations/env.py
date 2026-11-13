"""Alembic environment that requires an explicit PostgreSQL URL."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from ai_learning_rag.database import DatabaseSettings
from ai_learning_rag.database_models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = DatabaseSettings.from_env()
config.set_main_option("sqlalchemy.url", settings.url.render_as_string(hide_password=False))
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL without connecting; the URL is still explicit."""

    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        version_table_schema="rag",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run reviewed migrations without application-startup coupling."""

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            version_table_schema="rag",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
