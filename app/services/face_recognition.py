from fastapi import HTTPException, UploadFile
from typing import List, Optional
from app.models.schemas import Face
import numpy as np
from collections import deque

class FaceRecognitionService:
    def __init__(self, face_analysis_model, face_mesh):
        self.face_analysis_model = face_analysis_model
        self.talking_centroids_history = deque(maxlen=150)
        self.trust_threshold = 0.01  # Maximum variance to consider location trustworthy
        self.talking_ratio_threshold = 0.25  # Mouth aspect ratio threshold for talking
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = face_mesh

    def _get_face_center(self, landmarks):
        """Calculate the center of a face from landmarks"""
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        return np.array([sum(xs)/len(xs), sum(ys)/len(ys)])

    def _calculate_mouth_aspect_ratio(self, landmarks):
        """Calculate mouth aspect ratio for talking detection"""
        try:
            lm13 = landmarks.landmark[13]  # Upper inner lip
            lm14 = landmarks.landmark[14]  # Lower inner lip
            lm78 = landmarks.landmark[78]  # Left mouth corner
            lm308 = landmarks.landmark[308]  # Right mouth corner
            
            mouth_height = ((lm13.x - lm14.x)**2 + (lm13.y - lm14.y)**2)**0.5
            mouth_width = ((lm78.x - lm308.x)**2 + (lm78.y - lm308.y)**2)**0.5
            return mouth_height / (mouth_width + 1e-6)  # Avoid division by zero
        except Exception:
            return 0

    def _update_speaker_tracking(self, frame):
        """Detect talking faces and update tracking history"""
        results = self.mp_face_mesh.process(frame)
        if not results.multi_face_landmarks:
            return
            
        current_talking_centers = []
        
        for face_landmarks in results.multi_face_landmarks:
            ratio = self._calculate_mouth_aspect_ratio(face_landmarks)
            if ratio > self.talking_ratio_threshold:
                center = self._get_face_center(face_landmarks)
                current_talking_centers.append(center)
        
        # Store the average of all talking faces in this frame
        if current_talking_centers:
            avg_center = np.mean(current_talking_centers, axis=0)
            self.talking_centroids_history.append(avg_center)

    def get_speaker_location(self):
        """
        Returns a dict with:
        - centroid: (x,y) average position of speaker(s)
        - is_trustworthy: bool indicating if the location is reliable
        """
        if not self.talking_centroids_history:
            return {'centroid': None, 'is_trustworthy': False}
            
        centroids = np.array(self.talking_centroids_history)
        avg_center = tuple(np.mean(centroids, axis=0))
        
        # Calculate variance to determine trustworthiness
        if len(centroids) < 10:  # Not enough data
            return {'centroid': avg_center, 'is_trustworthy': False}
            
        variance = np.var(centroids, axis=0).mean()
        is_trustworthy = variance < self.trust_threshold
        
        return {'centroid': avg_center, 'is_trustworthy': is_trustworthy}
    
    def embed(self, image: np.ndarray) -> List[float]:
        """
        Generates an embedding vector from the given image file.
        Raises HTTPException if no face or multiple faces are detected.
        """
        faces = self.face_analysis_model.get(image)
        if len(faces) > 1:
            raise HTTPException(
                status_code=400, detail="More than one face detected. Please upload an image with exactly one face.")
        if len(faces) < 1:
            raise HTTPException(
                status_code=400, detail="No face detected. Please upload an image with a clear face.")

        return faces[0].embedding.tolist()

    def identify(self, image: np.ndarray) -> List[Face]:
        """
        Generates an embedding vector/s from the given image file for identification.
        """
        self._update_speaker_tracking(image)
        
        faces = self.face_analysis_model.get(image)
        identified_faces = []
        for face in faces:
            identified_faces.append(Face(bbox=face.bbox.tolist(), embeddings=face.embedding.tolist() ))

        return identified_faces

    def identifySingleFace(self, image: np.ndarray) -> Optional[Face]:
        """
        Generates an embedding vector from the given image file for identification.
        Raises HTTPException if no face or multiple faces are detected.
        """
        faces = self.face_analysis_model.get(image)

        if len(faces) < 1:
            return None

        image_center = np.array(
            [image.shape[1] / 2, image.shape[0] / 2])

        closest_face = None
        min_distance = float('inf')

        for face in faces:
            x1, y1, x2, y2 = face.bbox
            face_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

            distance = abs(face_center[0] - image_center[0]) + \
                abs(face_center[1] - image_center[1])

            if distance < min_distance:
                min_distance = distance
                closest_face = face

        return Face(bbox=closest_face.bbox.tolist(), embeddings=closest_face.embedding.tolist()) 
