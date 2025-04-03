from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import json
from typing import Optional
from app.database.connection import get_db
from app.dependencies import get_face_recognition_service
from app.services.face_recognition import FaceRecognitionService
from app.utils.image_validation import validate_image_file
from app.database.crud import (
    save_embedding,
    update_embedding,
    has_embedding_conflict,
    find_closest_matches,
    find_closest_match_single_face
)
from app.api.v2.schemas import (
    EmbedResponseV2,
    IdentifyResponseV2
)
from app.models.schemas import (
    Match
)
from app.api.v2.tracker import SimpleFaceTracker

import logging

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, message: dict):
        await websocket.send_json(message)


manager = ConnectionManager()


async def process_image_frame(image_data: bytes, db, face_service):
    """Helper function to process an image frame"""
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode image")

    return frame


@router.post("/embed", response_model=str)
async def embed_face(
    person_id: str = Form(...),
    image: UploadFile = File(...),
    db=Depends(get_db),
    face_service: FaceRecognitionService = Depends(
        get_face_recognition_service)
):

    # Validate that the uploaded file is an image
    validate_image_file(image)

    try:
        image_data = await image.read()

        image_array = np.frombuffer(image_data, np.uint8)

        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if decoded_image is None:
            raise HTTPException(
                status_code=400, detail="Could not decode image.")
        embedding = face_service.embed(decoded_image)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embedding: {str(e)}")

    try:
        if (embedding):
            if (has_embedding_conflict(db, person_id, embedding)):
                logger.error(
                    f"Embedding Conflict detected for Person ID: {person_id}")
                return JSONResponse(
                    status_code=409,
                    content={
                        "detail": f"Face embedding conflicts with existing entry for person: {person_id}"}
                )
            save_embedding(db, person_id, embedding)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save embedding: {str(e)}")

    return person_id if embedding else "Unknown"


@router.post("/update", response_model=str)
async def update_face(
    person_id: str = Form(...),
    image: UploadFile = File(...),
    db=Depends(get_db),
    face_service: FaceRecognitionService = Depends(
        get_face_recognition_service)
):

    # Validate that the uploaded file is an image
    validate_image_file(image)

    try:
        image_data = await image.read()

        image_array = np.frombuffer(image_data, np.uint8)

        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if decoded_image is None:
            raise HTTPException(
                status_code=400, detail="Could not decode image.")
        embedding = face_service.embed(decoded_image)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embedding: {str(e)}")

    try:
        if (embedding):
            update_embedding(db, person_id, embedding)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save embedding: {str(e)}")

    return person_id if embedding else "Unknown"


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
            data = await websocket.receive()

            try:
                speaker_location = face_service.get_speaker_location()
                if speaker_location:
                    pass
                    # print(f"Probable speaker at: {speaker_location}")
            except Exception as e:
                await manager.send_json(websocket, {
                    "status": "error",
                    "error": f"Connection error: {str(e)}"
                })
                manager.disconnect(websocket)
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
            else:
                image_bytes = data["bytes"]

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
                    lip_center=speaker_location["centroid"],
                    status="success"
                ).dict()

                await manager.send_json(websocket, response)

            except Exception as e:
                await manager.send_json(websocket, {
                    "status": "error",
                    "error": str(e)
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await manager.send_json(websocket, {
            "status": "error",
            "error": f"Connection error: {str(e)}"
        })
        manager.disconnect(websocket)


@router.websocket("/identify/single")
async def identify_single_face_ws(
    websocket: WebSocket,
    db=Depends(get_db),
    face_service: FaceRecognitionService = Depends(
        get_face_recognition_service)
):
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive()

            if "text" in data:
                message = json.loads(data["text"])
                if message.get("action") == "close":
                    break

                image_bytes = base64.b64decode(message["image"])
            else:
                image_bytes = data["bytes"]

            try:
                frame = await process_image_frame(image_bytes, db, face_service)
                identified_face = face_service.identifySingleFace(frame)
                match = find_closest_match_single_face(
                    db, identified_face) if identified_face is not None else None

                response = IdentifyResponseV2(matches=[match] if match else [],
                                              face_detected=match is not None,
                                              processed_faces=1,
                                              status="success").dict()
                await manager.send_json(websocket, response)

            except Exception as e:
                await manager.send_json(websocket, {
                    "status": "error",
                    "error": str(e)
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await manager.send_json(websocket, {
            "status": "error",
            "error": f"Connection error: {str(e)}"
        })
        manager.disconnect(websocket)
