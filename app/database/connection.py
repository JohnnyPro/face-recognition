import os
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import Depends
from .setup import initialize_database
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
db_username = os.getenv('DATABASE_USERNAME')
db_password = os.getenv('DATABASE_PASSWORD')
db_host = os.getenv('DATABASE_HOST')
db_port = os.getenv('DATABASE_PORT')
db_name = os.getenv('DATABASE_NAME')
# Run database setup on first application start
initialize_database()


def get_db():
    db = psycopg2.connect(
        dbname=os.getenv("DATABASE_NAME", "face_recognition"),
        user=os.getenv("DATABASE_USERNAME", "postgres"),
        password=os.getenv("DATABASE_PASSWORD", "postgress"),
        host=os.getenv("DATABASE_HOST", "db"),
        port=os.getenv("DATABASE_PORT", "5432"),
        cursor_factory=RealDictCursor
    )
    try:
        yield db
    finally:
        db.close()