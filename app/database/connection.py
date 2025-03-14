import os
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import Depends
from .setup import initialize_database  

# Run database setup on first application start
initialize_database()


def get_db():
    db = psycopg2.connect(
         dbname=os.getenv("POSTGRES_DB", "face_recognition"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgress"),
        host=os.getenv("POSTGRES_HOST", "db"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        cursor_factory=RealDictCursor
    )
    try:
        yield db
    finally:
        db.close()