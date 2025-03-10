from typing import List, Optional, Tuple
from ..models.schemas import Person

THRESHOLD = 0.45


def create_person(db, name: str) -> Optional[Person]:
    query = "INSERT INTO people (name) VALUES (%s) RETURNING id, name"
    with db.cursor() as cursor:
        cursor.execute(query, (name,))
        result = cursor.fetchone()
        if result:
            return Person(id=result["id"], name=result["name"])
        return None


def save_embedding(db, person_id: int, embedding: List[float]):
    query = "INSERT INTO embeddings (person_id, embedding) VALUES (%s, %s)"
    with db.cursor() as cursor:
        cursor.execute(query, (person_id, embedding))
        db.commit()


def find_closest_matches(db, embeddings: List[List[float]]) -> List[Tuple[Person, float]]:
    query = """
        SELECT p.id, p.name, f.embedding <=> %s::vector AS distance
        FROM people p
        JOIN embeddings f ON p.id = f.person_id
        WHERE f.embedding <=> %s::vector < %s
        ORDER BY distance
        LIMIT 1;
    """
    results = []
    with db.cursor() as cursor:
        for embedding in embeddings:
            cursor.execute(query, (embedding, embedding, THRESHOLD))
            result = cursor.fetchone()
            if result is not None:
                person_id = result["id"]
                name = result["name"]
                confidence = 1 - result["distance"]
                results.append((Person(id=person_id, name=name), confidence))
            else:
                results.append((Person(id=0, name="Unknown"), 0))

        return results
