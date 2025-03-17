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
            raise HTTPException(
                status_code=400, detail="No face detected. Please upload an image with a clear face.")

        return faces[0].embedding.tolist()

    def identify(self, image: np.ndarray) -> List[List[float]]:
        """
        Generates an embedding vector/s from the given image file for identification.
        """
        faces = self.face_analysis_model.get(image)
        embeddings = []
        for face in faces:
            embeddings.append(face.embedding.tolist())

        return embeddings

    def identifySingleFace(self, image: np.ndarray) -> List[float]:
        """
        Generates an embedding vector from the given image file for identification.
        Raises HTTPException if no face or multiple faces are detected.
        """
        faces = self.face_analysis_model.get(image)

        if len(faces) < 1:
            raise HTTPException(
                status_code=400, detail="No face detected. Please upload an image with a clear face.")

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

        return closest_face.embedding.tolist()
