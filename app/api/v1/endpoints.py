from fastapi import APIRouter, Depends, HTTPException, Form, File, UploadFile
from app.models.schemas import EmbedResponse, IdentifyResponse
from app.database.crud import save_embedding, find_closest_matches, find_closest_match_single_face
from app.database.connection import get_db
from app.dependencies import get_face_recognition_service
from app.utils.image_validation import validate_image_file
from app.services.face_recognition import FaceRecognitionService
import cv2
import numpy as np
import os
from typing import Tuple


router = APIRouter()


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
        embedding = face_service.embed_static(decoded_image)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embedding: {str(e)}")

    try:
        save_embedding(db, person_id, embedding)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save embedding: {str(e)}")

    return person_id


@router.post("/identify", response_model=IdentifyResponse)
async def identify_faces(
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
        identified_faces = face_service.identify(decoded_image)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embedding: {str(e)}")

    try:
        matches = find_closest_matches(db, identified_faces)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to identify face: {str(e)}")

    return IdentifyResponse(matches=matches, face_detected=len(matches) != 0)


@router.post("/identify-face", response_model=IdentifyResponse)
async def identifySingleFace(
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

        identified_face = face_service.identifySingleFace(decoded_image)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embedding: {str(e)}")

    try:
        match = find_closest_match_single_face(db, identified_face)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to identify face: {str(e)}")

    return IdentifyResponse(matches=[match], face_detected=match is not None)
