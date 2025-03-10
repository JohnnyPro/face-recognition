from fastapi import HTTPException, UploadFile
from typing import List
import numpy as np


class FaceRecognitionService:
    def __init__(self, face_analysis_model):
        self.face_analysis_model = face_analysis_model

    def embed(self, image: np.ndarray) -> List[float]:
        """
        Generates an embedding vector from the given image file.
        Raises HTTPException if no face or multiple faces are detected.
        """
        print(image.shape)
        faces = self.face_analysis_model.get(image)
        if len(faces) > 1:
            raise HTTPException(
                status_code=400, detail="More than one face detected. Please upload an image with exactly one face.")
        if len(faces) < 1:
            raise HTTPException(status_code=400, detail="No face detected. Please upload an image with a clear face.")

        return faces[0].embedding.tolist()

    def identify(self, image: np.ndarray) -> List[List[float]]:
        """
        Generates an embedding vector from the given image file for identification.
        Raises HTTPException if no face or multiple faces are detected.
        """
        faces = self.face_analysis_model.get(image)
        embeddings = []
        for face in faces:
            embeddings.append(face.embedding.tolist())

        return embeddings
