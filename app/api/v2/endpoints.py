from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState
import cv2
import numpy as np
import base64
import json
from typing import Optional, List
from app.database.connection import get_db
from app.dependencies import get_face_recognition_service
from app.services.face_recognition import FaceRecognitionService
from app.utils.image_validation import validate_image_file
from app.database.crud import (
    save_embedding,
    update_embedding,
    has_embedding_conflict,
    is_duplicate_seed,
    find_closest_matches,
    find_closest_match_single_face
)
from app.api.v2.schemas import (
    EmbedResponseV2,
    IdentifyResponseV2,
    SeedResponse
)
from app.models.schemas import (
    Match
)
from app.api.v2.tracker import SimpleFaceTracker
from app.api.v2.face_seen_tracker import MatchTimeTracker

import logging

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

face_seen_tracker = MatchTimeTracker()


class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, message: dict):
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)
            else:
                logger.info(websocket.client_state)
                logger.warning("Attempted to send to a closed WebSocket, skipping send")
        except Exception as e:
            logger.error(f"Failed to send JSON message: {e}")


manager = ConnectionManager()


async def process_image_frame(image_data: bytes, db, face_service):
    """Helper function to process an image frame"""
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode image")

    return frame


@router.post("/embed")
async def embed_face(
    person_id: str = Form(...),
    image: UploadFile = File(...),
    db=Depends(get_db),
    face_service: FaceRecognitionService = Depends(
        get_face_recognition_service)
):

    try:
        # Validate that the uploaded file is an image
        validate_image_file(image)
    except Exception as e:
        logger.error(f"Image validation failed for {image.filename}: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )
    embedding = None
    try:
        image_data = await image.read()
        image_array = np.frombuffer(image_data, np.uint8)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if decoded_image is None:
            logger.error(f"Could not decode image file: {image.filename}")
            return JSONResponse(
                status_code=400,
                content={"status": "error",
                         "message": "Could not decode image."}
            )

        embedding = face_service.embed(decoded_image)

    # Catch specific exceptions from the service if needed, otherwise general Exception
    except Exception as e:
        logger.error(f"Error generating embedding for {person_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error",
                     "message": f"Error generating embedding: {str(e)}"}
        )

    if embedding is None or not embedding:
        logger.warning(
            f"No face detected or embedding could not be generated for Person ID: {person_id}, Image: {image.filename}")
        return JSONResponse(
            status_code=400,
            content={"status": "error",
                     "message": "No face detected or embedding could not be generated from the provided image."}
        )

    try:
        if has_embedding_conflict(db, person_id, embedding):
            logger.error(
                f"Embedding Conflict detected for Person ID: {person_id}")
            return JSONResponse(
                status_code=409,  # Conflict
                content={
                    "status": "conflict",
                    "message": f"Face embedding conflicts with an existing entry."
                }
            )

        # If no conflict, save the embedding
        save_embedding(db, person_id, embedding)
        logger.info(f"Successfully saved embedding for Person ID: {person_id}")

        return JSONResponse(
            status_code=201,
            content={
                "status": "success",
                "message": f"Embedding saved successfully for person ID: {person_id}",
                "person_id": person_id
            }
        )

    except Exception as e:
        logger.error(
            f"Failed to check conflict or save embedding for {person_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error",
                     "message": f"Failed to save embedding or check for conflicts: {str(e)}"}
        )


@router.post("/seed")
async def seed_face(
    person_id: str = Form(...),
    image: UploadFile = File(...),
    db=Depends(get_db),
    face_service: FaceRecognitionService = Depends(
        get_face_recognition_service)
):

    
    try:
        validate_image_file(image)
    except Exception as e:
        logger.error(f"Image validation failed for {image.filename}: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"status": "error",
                     "message": f"Invalid image file: {str(e)}"}
        )

    embedding = None
    try:
        image_data = await image.read()
        image_array = np.frombuffer(image_data, np.uint8)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if decoded_image is None:
            logger.error(f"Could not decode image file: {image.filename}")
            return JSONResponse(
                status_code=400,
                content={"status": "error",
                         "message": "Could not decode image."}
            )

        embedding = face_service.embed_static(decoded_image)

    except Exception as e:
        logger.error(
            f"Error generating static embedding for seeding {person_id}: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error",
                     "message": f"Error generating embedding: {str(e)}"}
        )

    # 3. Check if embedding was generated successfully
    if embedding is None or not embedding:
        logger.warning(
            f"No face detected or static embedding could not be generated during seeding for Person ID: {person_id}, Image: {image.filename}")
        return JSONResponse(
            status_code=400,  # Bad Request
            content={"status": "error",
                     "message": "No face detected or embedding could not be generated from the provided image for seeding."}
        )

    try:
        if is_duplicate_seed(db, person_id, embedding):
            logger.warning(
                f"Duplicate seed detected for Person ID: {person_id}")
            response_content = SeedResponse(
                status="failed",
                error=409,
                already_seeded=True,
                person_id=person_id,
                message="Duplicate seed detected. Person already seeded with a similar face.",
            )

            return JSONResponse(
                status_code=409,
                content=response_content.dict()
            )

        elif has_embedding_conflict(db, person_id, embedding):
            logger.info(
                f"Updating embedding for Person ID: {person_id}")
            update_embedding(db, person_id, embedding)
            response_content = SeedResponse(
                status="success",
                already_seeded=False,
                person_id=person_id,
                message="Updated existing embedding.",
            )
            return JSONResponse(
                status_code=200,
                content=response_content.dict()
            )

        else:
            logger.info(
                f"Creating new seed embedding for Person ID: {person_id}")
            save_embedding(db, person_id, embedding)
            response_content = SeedResponse(
                status="success",
                already_seeded=False,
                person_id=person_id,
                message="Created new seed embedding.",
            )

            return JSONResponse(
                status_code=201,
                content=response_content.dict()
            )

    except Exception as e:
        # Catch errors specifically from the database/seeding logic part
        # (is_duplicate_seed, has_embedding_conflict, update_embedding, save_embedding)
        logger.error(
            f"Failed during seeding database operation for {person_id}: {str(e)}")
        
        return JSONResponse(
            status_code=400,
            content={"status": "error",
                     "message": f"Failed to process seed request due to database or internal error: {str(e)}"}
        )

    logger.error(
        f"Reached unexpected end of seeding function for Person ID: {person_id}")
    return JSONResponse(
        status_code=500,
        content={"status": "error",
                 "message": "An unexpected server error occurred during seeding."}
    )


@router.post("/mark-greeted")
async def mark_greeted(
    person_ids: List[str] = Form(...),
    db=Depends(get_db),
    face_service: FaceRecognitionService = Depends(
        get_face_recognition_service),
):
    try:
        for person_id in person_ids:
            face_seen_tracker.mark_greeted(person_id)

        return {"status_code": 200, "status": "success", "marked_greeted": person_ids}
    except Exception as e:
        logger.error(
            f"Error marking greeted for persons: {person_ids}. Error: {str(e)}")
        return {"status_code": 500, "status": "error", "message": str(e)}



@router.post("/update")
async def update_face(
    person_id: str = Form(...),
    image: UploadFile = File(...),
    db=Depends(get_db),
    face_service: FaceRecognitionService = Depends(
        get_face_recognition_service)
):

    try:
        validate_image_file(image)
    except Exception as e:
        logger.error(f"Image validation failed for {image.filename}: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Invalid image file: {str(e)}"}
        )

    embedding = None
    try:
        image_data = await image.read()
        image_array = np.frombuffer(image_data, np.uint8)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if decoded_image is None:
            logger.error(f"Could not decode image file: {image.filename}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Could not decode image."}
            )

        embedding = face_service.embed(decoded_image)

    except Exception as e:
        logger.error(f"Error generating embedding for update, Person ID {person_id}: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": f"Error generating embedding: {str(e)}"}
        )

    if embedding is None or not embedding:
         logger.warning(f"No face detected or embedding could not be generated for update, Person ID: {person_id}, Image: {image.filename}")
         return JSONResponse(
             status_code=400,
             content={"status": "error", "message": "No face detected or embedding could not be generated from the provided image for update."}
         )

    try:
        update_embedding(db, person_id, embedding)
        logger.info(f"Successfully updated embedding for Person ID: {person_id}")

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Embedding updated successfully for person ID: {person_id}",
                "person_id": person_id
            }
        )

    except Exception as e:
        logger.error(f"Failed to update embedding for {person_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to update embedding: {str(e)}"}
        )

    logger.error(f"Reached unexpected end of update function for Person ID: {person_id}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "An unexpected server error occurred during update."}
    )

@router.websocket("/identify")
async def identify_faces_ws(
    websocket: WebSocket,
    db=Depends(get_db),
    face_service: FaceRecognitionService = Depends(
        get_face_recognition_service)
):
    await manager.connect(websocket)

    # Initialize simple tracker for this connection
    tracker = SimpleFaceTracker(
        iou_threshold=0.45, max_missed_frames=30)  # 30 frame tolerance

    try:
        while True:
            if websocket.client_state == WebSocketState.DISCONNECTED:
                break
            elif websocket.client_state == WebSocketState.CONNECTING:
                continue
            
            data = await websocket.receive()

            try:
                speaker_location = face_service.get_speaker_location()
            except Exception as e:
                speaker_location = None
                logger.info("Speaker Location retrieval failed")
            # Handle configuration message
            if "text" in data:
                message = json.loads(data["text"])
                if message.get("action") == "close":
                    logger.info(f"Closing WebSocket Connection")
                    break

                if message.get("action") == "configure":
                    websocket.threshold = message.get("threshold", 0.5)
                    websocket.max_faces = message.get("max_faces", 5)
                    logger.info(
                        f"Configured Face Recognition Module for Threshold: {websocket.threshold} and Maximum Faces: {websocket.max_faces}")
                    continue

                image_bytes = base64.b64decode(message["image"])
            
            elif "bytes" in data:
                image_bytes = data["bytes"]
            else:
                # Neither 'text' nor 'bytes' found - unexpected format
                logger.warning(f"Received unexpected WebSocket data format: {data.keys()}")
                try:
                    await manager.send_json(websocket, {"status": "error", "error": "Unsupported data format received."})
                except Exception as send_error:
                    logger.error(f"Failed to send unsupported format error to client: {send_error}")
                continue

            if image_bytes is None:
                await manager.send_json(websocket, {"status": "error", "error": "No image data provided"})
                continue


            try:
                frame = await process_image_frame(image_bytes, db, face_service)
                identified_faces = face_service.identify(frame)

                threshold = getattr(websocket, "threshold", 0.5)
                max_faces = getattr(websocket, "max_faces", 5)

                # Get matches from current frame (may contain Unknowns)
                matches = find_closest_matches(
                    db, identified_faces,
                    threshold=threshold,
                    max_results=max_faces
                )

                # Get tracked faces from previous frame BEFORE updating
                previous_tracks = tracker.get_previous_frame_tracks()

                # Now update tracker with current frame's matches
                tracker.update_tracks(matches)
                # Update last seen data with current frame's matches
                face_seen_tracker.update(matches)
                new_faces = face_seen_tracker.get_new_faces()
                # Combine results - this is where we use tracking to fill in Unknowns
                final_matches = []
                used_track_ids = set()
                only_tracked_ids = set()

                # First process all recognized faces
                for match in matches:
                    if match.person_id != "Unknown":
                        final_matches.append(match)
                        used_track_ids.add(match.person_id)
                    else:
                        # Try to match this Unknown with a previous track
                        best_track = None
                        best_iou = 0

                        for track in previous_tracks:
                            iou = SimpleFaceTracker.calculate_iou(
                                track["bbox"], match.bbox)
                            if iou > best_iou and iou >= 0.45:  # Matching threshold
                                best_iou = iou
                                best_track = track

                        if best_track:
                            # Use the tracked identity but with reduced confidence
                            final_matches.append(Match(
                                person_id=best_track["person_id"],
                                confidence=best_track["confidence"]*0.8,
                                # confidence=.12,
                                bbox=match.bbox
                            ))
                            used_track_ids.add(best_track["person_id"])
                            only_tracked_ids.add(best_track["person_id"])
                        else:
                            # Keep as Unknown if no good track match
                            final_matches.append(match)

                # Limit to max_faces and sort by confidence
                final_matches = sorted(
                    final_matches, key=lambda x: -x.confidence)[:max_faces]
                # logger.info(f"Final Matches: {[m.dict() for m in final_matches]}")
                response = IdentifyResponseV2(
                    matches=final_matches,
                    face_detected=len(final_matches) > 0,
                    processed_faces=len(identified_faces),
                    tracked=only_tracked_ids,
                    # lip_center=speaker_location["centroid"] if speaker_location["is_trustworthy"] else [],
                    lip_center=speaker_location["centroid"] if speaker_location else [],
                    new_faces=new_faces,
                    status="success"
                ).dict()

                await manager.send_json(websocket, response)

            except Exception as e:
                await manager.send_json(websocket, {
                    "status": "error",
                    "error": str(e)
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        logger.info("Unexpected error that's not disconnect")
        await manager.send_json(websocket, {
            "status": "error",
            "error": f"Connection error: {str(e)}"
        })
        manager.disconnect(websocket)
    
    finally:
        manager.disconnect(websocket)



# @router.websocket("/identify/single")
# async def identify_single_face_ws(
#     websocket: WebSocket,
#     db=Depends(get_db),
#     face_service: FaceRecognitionService = Depends(
#         get_face_recognition_service)
# ):
#     await manager.connect(websocket)

#     try:
#         while True:
#             data = await websocket.receive()

#             if "text" in data:
#                 message = json.loads(data["text"])
#                 if message.get("action") == "close":
#                     break

#                 image_bytes = base64.b64decode(message["image"])
#             else:
#                 image_bytes = data["bytes"]

#             try:
#                 frame = await process_image_frame(image_bytes, db, face_service)
#                 identified_face = face_service.identifySingleFace(frame)
#                 match = find_closest_match_single_face(
#                     db, identified_face) if identified_face is not None else None

#                 response = IdentifyResponseV2(matches=[match] if match else [],
#                                               face_detected=match is not None,
#                                               processed_faces=1,
#                                               status="success").dict()
#                 await manager.send_json(websocket, response)

#             except Exception as e:
#                 await manager.send_json(websocket, {
#                     "status": "error",
#                     "error": str(e)
#                 })

#     except WebSocketDisconnect:
#         manager.disconnect(websocket)
#     except Exception as e:
#         await manager.send_json(websocket, {
#             "status": "error",
#             "error": f"Connection error: {str(e)}"
#         })
#         manager.disconnect(websocket)
