from typing import List, Optional, Tuple, DefaultDict
from app.models.schemas import Match, Face
THRESHOLD = 0.45

def save_embedding(db, person_id: int, embedding: List[float]):
    query = "INSERT INTO embeddings (person_id, embedding) VALUES (%s, %s)"
    with db.cursor() as cursor:
        cursor.execute(query, (person_id, embedding))
        db.commit()


def find_closest_matches(db, faces: List[Face], threshold=THRESHOLD, max_results=5) -> List[Match]:
    query = """
        SELECT person_id, embedding <=> %s::vector AS distance
        FROM embeddings WHERE embedding <=> %s::vector < %s
        ORDER BY distance
        LIMIT 1;
    """
    results = []
    with db.cursor() as cursor:
        for face in faces:
            embedding = face.embeddings
            cursor.execute(query, (embedding, embedding, threshold))
            result = cursor.fetchone()
            if result is not None:
                person_id = result["person_id"]
                confidence = 1 - result["distance"]
                results.append(Match(person_id=person_id, confidence=confidence, bbox=face.bbox))
            else:
                results.append(Match(person_id="Unknown", confidence=0, bbox=face.bbox))
                
            if len(results) >= max_results:
                break 

        return results


def find_closest_match_single_face(db, face: Face) -> Match:
    closest_match = find_closest_matches(db, [face])
    return closest_match[0]