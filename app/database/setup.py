import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.errors import DuplicateDatabase, DuplicateTable
import logging
import os

# Retrieve environment variables
db_username = os.getenv('DATABASE_USERNAME')
db_password = os.getenv('DATABASE_PASSWORD')
db_host = os.getenv('DATABASE_HOST')
db_port = os.getenv('DATABASE_PORT')
db_name = os.getenv('DATABASE_NAME')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_database():
    """Ensure the database exists before proceeding."""
    try:
        conn = psycopg2.connect(
            dbname=db_name, user=db_username, password=db_password, host=db_host, port=db_port
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Check if the database already exists
        cursor.execute(
            f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()

        if not exists:
            logger.info(f"Creating database: {db_name}")
            cursor.execute(f"CREATE DATABASE {db_name}")
        else:
            logger.info(f"Database '{db_name}' already exists.")

        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.close()
        conn.close()
    except DuplicateDatabase:
        logger.info(f"Database '{db_name}' already exists.")
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise


def setup_tables():
    """Ensure required tables exist."""
    try:
        conn = psycopg2.connect(
            dbname=db_name, user=db_username, password=db_password, host=db_host, port=db_port
        )
        cursor = conn.cursor()

        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("pgvector extension enabled.")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                person_id TEXT,
                embedding VECTOR(512) NOT NULL
            )
        """)

        conn.commit()
        logger.info("Tables created successfully.")

        cursor.close()
        conn.close()
    except DuplicateTable:
        logger.info("Tables already exist.")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise


def initialize_database():
    """Run database setup on application start."""
    create_database()
    setup_tables()