import os
from dataclasses import dataclass
from typing import Any

import pandas as pd


TABLE_NAME = "customer_data"


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: str
    name: str
    user: str
    password: str

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


def load_db_config() -> DatabaseConfig:
    """Load PostgreSQL configuration from environment variables."""
    values = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT", "5432"),
        "name": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }
    missing = [key.upper().replace("NAME", "DB_NAME") for key, value in values.items() if not value]
    if missing:
        pretty_names = {
            "HOST": "DB_HOST",
            "PORT": "DB_PORT",
            "DB_NAME": "DB_NAME",
            "USER": "DB_USER",
            "PASSWORD": "DB_PASSWORD",
        }
        missing = [pretty_names.get(item, item) for item in missing]
        raise ValueError(f"Missing database environment variables: {', '.join(missing)}")

    return DatabaseConfig(
        host=str(values["host"]),
        port=str(values["port"]),
        name=str(values["name"]),
        user=str(values["user"]),
        password=str(values["password"]),
    )


def get_engine():
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise RuntimeError(
            "SQLAlchemy is not installed. Run: pip install -r requirements.txt"
        ) from exc

    config = load_db_config()
    return create_engine(config.url, pool_pre_ping=True)


def test_connection(engine) -> None:
    from sqlalchemy import text

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))


def replace_customer_data(df: pd.DataFrame, engine) -> int:
    """Replace the uploaded dataset in PostgreSQL."""
    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False, method="multi", chunksize=1000)
    return len(df)


def fetch_dataset_analytics(engine) -> dict[str, Any]:
    from sqlalchemy import text, inspect

    try:
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns(TABLE_NAME)]
    except Exception:
        columns = []

    avg_income_sql = "AVG(income)" if "income" in columns else "NULL"
    avg_loan_sql = "AVG(loan_amount)" if "loan_amount" in columns else "NULL"

    query = text(
        f"""
        SELECT
            COUNT(*) AS total_customers,
            {avg_income_sql} AS average_income,
            {avg_loan_sql} AS average_loan_amount
        FROM {TABLE_NAME}
        """
    )
    with engine.connect() as connection:
        row = connection.execute(query).mappings().one()
    return dict(row)


def process_upload_to_database(df: pd.DataFrame) -> tuple[int, dict[str, Any]]:
    """Connect, replace customer_data, and return stored row count plus analytics."""
    try:
        from sqlalchemy.exc import SQLAlchemyError
    except ImportError as exc:
        raise RuntimeError(
            "SQLAlchemy is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        engine = get_engine()
        test_connection(engine)
        rows_stored = replace_customer_data(df, engine)
        analytics = fetch_dataset_analytics(engine)
        return rows_stored, analytics
    except (SQLAlchemyError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
