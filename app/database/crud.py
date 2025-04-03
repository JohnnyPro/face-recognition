from typing import List, Optional, Tuple, DefaultDict
from app.models.schemas import Match, Face
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


THRESHOLD = 0.5


def is_duplicate_seed(db, person_id, embedding) -> bool:
    logger.info("Checking for duplicate seed...")
    query = """
        SELECT COUNT(*) > 0 AS exists
        FROM embeddings
        WHERE embedding <=> %s::vector < %s
          AND person_id = %s;
    """
    with db.cursor() as cursor:
        cursor.execute(query, (embedding, 0.02, person_id))
        return bool(cursor.fetchone()["exists"])

def has_embedding_conflict(db, person_id, embedding) -> bool:
    query = """
        SELECT COUNT(*) > 0 AS exists
        FROM embeddings
        WHERE embedding <=> %s::vector < %s
          AND person_id != %s;
    """
    with db.cursor() as cursor:
        cursor.execute(query, (embedding, THRESHOLD, person_id))
        return bool(cursor.fetchone()["exists"])


def save_embedding(db, person_id: int, embedding: List[float]):
    query = "INSERT INTO embeddings (person_id, embedding) VALUES (%s, %s)"
    with db.cursor() as cursor:
        cursor.execute(query, (person_id, embedding))
        db.commit()


def update_embedding(db, person_id: int, embedding: List[float], threshold=THRESHOLD):
    """
   Update embeddings by:
   1. Finding all similar faces (under threshold) and updating their person_id
   2. Inserting the new embedding with the specified person_id

   Args:
       db: Database connection
       person_id: The target person_id to assign to matching faces
       embedding: The new embedding vector to insert and use for similarity matching
   """
    query = """
        UPDATE embeddings
        SET person_id = %s
        WHERE embedding <=> %s::vector < %s;
    """
    find_matches_query = """
        SELECT person_id, embedding <=> %s::vector AS distance
        FROM embeddings WHERE embedding <=> %s::vector < %s
        ORDER BY distance;
    """

    logger.debug(
        f"Attempting to update embeddings similar to new embedding for person_id {person_id} (threshold: {threshold})")

    try:
        with db.cursor() as cursor:
            cursor.execute(find_matches_query,
                           (embedding, embedding, threshold))
            matches = cursor.fetchall()
            logger.info("Matches to be updated:")
            for match in matches:
                logger.info(
                    f"Person ID: {match['person_id']}, Distance: {match['distance']}")
        with db.cursor() as cursor:
            params = (person_id, embedding, threshold)
            cursor.execute(query, params)
            updated_row_count = cursor.rowcount  # Number of rows affected by the UPDATE
        db.commit()
        if updated_row_count > 0:
            logger.info(
                f"Successfully updated {updated_row_count} existing embeddings to person_id {person_id} based on similarity.")
        else:
            logger.info(
                f"No existing embeddings found within threshold {threshold} to update for person_id {person_id}.")

    except Exception as e:
        logger.exception(
            f"Unexpected error while updating similar embeddings for person ID {person_id}: {e}")
        db.rollback()
        updated_row_count = -1

    return updated_row_count


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
                results.append(Match(person_id=person_id,
                               confidence=confidence, bbox=face.bbox))
            else:
                results.append(Match(person_id="Unknown",
                               confidence=0, bbox=face.bbox))

            if len(results) >= max_results:
                break

        return results


def find_closest_match_single_face(db, face: Face) -> Match:
    closest_match = find_closest_matches(db, [face])
    return closest_match[0]
