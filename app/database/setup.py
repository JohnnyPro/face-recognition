import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.errors import DuplicateDatabase, DuplicateTable
import logging

# Database configuration
DB_NAME = "face_recognition"
DB_USER = "postgres"
DB_PASSWORD = "postgress"
DB_HOST = "postgres_db"
DB_PORT = "5432"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_database():
    """Ensure the database exists before proceeding."""
    try:
        conn = psycopg2.connect(
            dbname="postgres", user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Check if the database already exists
        cursor.execute(
            f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cursor.fetchone()

        if not exists:
            logger.info(f"Creating database: {DB_NAME}")
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
        else:
            logger.info(f"Database '{DB_NAME}' already exists.")

        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.close()
        conn.close()
    except DuplicateDatabase:
        logger.info(f"Database '{DB_NAME}' already exists.")
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise


def setup_tables():
    """Ensure required tables exist."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
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