from insightface.app import FaceAnalysis  # Replace with your actual import
from app.services.face_recognition import FaceRecognitionService
from fastapi import Depends
import mediapipe as mp

# Initialize the FaceAnalysis model once
face_analysis_model = FaceAnalysis(name='buffalo_sc')
face_analysis_model.prepare(ctx_id=-1, det_size=(640, 640))
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_recognition_service = FaceRecognitionService(face_analysis_model, face_mesh)


def get_face_recognition_service() -> FaceRecognitionService:
    """
    Dependency that returns the singleton instance of FaceRecognitionService.
    """
    return face_recognition_service
