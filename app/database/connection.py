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
        dbname=db_name,
        user=db_username,
        password=db_password,
        host=db_host,
        port=db_port,
        cursor_factory=RealDictCursor
    )
    try:
        yield db
    finally:
        db.close()
