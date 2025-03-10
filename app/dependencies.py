from insightface.app import FaceAnalysis  # Replace with your actual import
from app.services.face_recognition import FaceRecognitionService
from fastapi import Depends

# Initialize the FaceAnalysis model once
face_analysis_model = FaceAnalysis(name='buffalo_sc')
face_analysis_model.prepare(ctx_id=-1, det_size=(640, 640))

face_recognition_service = FaceRecognitionService(face_analysis_model)

def get_face_recognition_service() -> FaceRecognitionService:
    """
    Dependency that returns the singleton instance of FaceRecognitionService.
    """
    return face_recognition_service