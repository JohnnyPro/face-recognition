from fastapi import APIRouter, Depends, HTTPException, Form, File, UploadFile
from app.models.schemas import EmbedResponse, IdentifyResponse
from app.database.crud import save_embedding, find_closest_matches, find_closest_match_single_face
from app.database.connection import get_db
from app.dependencies import get_face_recognition_service
from app.utils.image_validation import validate_image_file
from app.services.face_recognition import FaceRecognitionService
from app.services.lip_movement import LipMovementDetector
from app.services.box_overlap import boxes_overlap
import cv2
import numpy as np
import os
from typing import Tuple, List
lip_detector = LipMovementDetector()


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
        embedding = face_service.embed(decoded_image)
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


@router.post("/detect-speaking-person", response_model=IdentifyResponse)
async def detect_speaking_person(
    images: List[UploadFile] = File(...),
    db=Depends(get_db),
    face_service: FaceRecognitionService = Depends(get_face_recognition_service)
):
    """
    This endpoint accepts a list of image frames (from the front end) and:
      1. Processes them using the LipMovementDetector to aggregate talking state.
      2. Uses face recognition (InsightFace) on one representative frame to obtain recognized faces with bounding boxes.
      3. Uses spatial correlation (bounding box overlap) to determine which recognized face is talking.
      4. Returns the identified person’s details (or “unknown” if no match).
    """
    frames = []
    for image in images:
        validate_image_file(image)
        try:
            image_data = await image.read()
            image_array = np.frombuffer(image_data, np.uint8)
            decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if decoded_image is None:
                raise HTTPException(status_code=400, detail="Could not decode image.")
            frames.append(decoded_image)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    if not frames:
        raise HTTPException(status_code=400, detail="No valid images provided.")

    # Process frames for lip movement detection.
    lip_results = lip_detector.process_frames(frames)
    # Find talking face(s) based on aggregated results.
    talking_face_ids = [face_id for face_id, info in lip_results.items() if info["talking"]]
    if not talking_face_ids:
        return IdentifyResponse(matches=[], face_detected=False)

    # For spatial correlation, choose one representative frame.
    rep_frame = frames[len(frames)//2]
    try:
        recognized_faces = face_service.identify(rep_frame)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error identifying faces: {str(e)}")

    # For each talking face from lip detection, check overlap with recognized faces.
    # Assume that recognized_faces has a bbox field.
    matched_persons = []
    for lip_face_id in talking_face_ids:
        lip_bbox = lip_results[lip_face_id]["bbox"]  # (x_min, y_min, x_max, y_max)
        for face in recognized_faces:
            face_bbox = face.bbox  # ensure this is in (x_min, y_min, x_max, y_max) format
            if boxes_overlap(lip_bbox, face_bbox):
                # Found a match, append the recognized face.
                matched_persons.append(face)
                break  # if one face matches, no need to check others

    # If no face overlaps, return unknown.
    if not matched_persons:
        return IdentifyResponse(matches=[], face_detected=False)
    
    # Return the matched recognized face(s).
    return IdentifyResponse(matches=matched_persons, face_detected=True)
