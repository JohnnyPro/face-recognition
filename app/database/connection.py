import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import Depends


def get_db():
    db = psycopg2.connect(
        dbname="face_recognition",
        user="postgres",
        password="postgress",
        cursor_factory=RealDictCursor
    )
    try:
        yield db
    finally:
        db.close()
