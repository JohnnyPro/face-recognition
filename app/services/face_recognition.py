from fastapi import HTTPException, UploadFile
from typing import List, Optional
from app.models.schemas import Face
import numpy as np
from collections import deque

class FaceRecognitionService:
    def __init__(self, face_analysis_model, face_mesh):
        self.face_analysis_model = face_analysis_model
        self.talking_centroids_history = deque(maxlen=90)
        self.trust_threshold = 0.02  # Maximum variance to consider location trustworthy
        self.talking_ratio_threshold = 0.12 # Mouth aspect ratio threshold for talking
        self.w = 100
        self.h = 100
        self.noTalkingFramesCounter = 0
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
        self.h, self.w, _ = frame.shape
        
        results = self.mp_face_mesh.process(frame)
        if not results.multi_face_landmarks:
            if self.noTalkingFramesCounter > 20 and len(self.talking_centroids_history) > 0:
                self.talking_centroids_history.clear()
            self.noTalkingFramesCounter += 1
            return
            
        current_talking_centers = []
        
        for face_landmarks in results.multi_face_landmarks:
            ratio = self._calculate_mouth_aspect_ratio(face_landmarks)
            if ratio > self.talking_ratio_threshold:
                center = self._get_face_center(face_landmarks)
                current_talking_centers.append(center)
        
        # Store the average of all talking faces in this frame
        if current_talking_centers:
            self.noTalkingFramesCounter = 0
            avg_center = np.mean(current_talking_centers, axis=0)
            self.talking_centroids_history.append(avg_center)
        else:
            if self.noTalkingFramesCounter > 20 and len(self.talking_centroids_history) > 0:
                self.talking_centroids_history.clear()
            self.noTalkingFramesCounter += 1
            
                

    def get_speaker_location(self):
        """
        Returns a dict with:
        - centroid: (x,y) average position of speaker(s)
        - is_trustworthy: bool indicating if the location is reliable
        """
        if not self.talking_centroids_history:
            return {'centroid': None, 'is_trustworthy': False}
            
        centroids = np.array(self.talking_centroids_history)
        num_frames = len(centroids)
            # Exponential weighting (higher weight for recent frames)
        alpha = 0.9  # Weight decay factor, adjust as needed
        weights = np.array([alpha ** (num_frames - i - 1) for i in range(num_frames)])
        weights /= weights.sum()  # Normalize weights

        weighted_avg = np.average(centroids, axis=0, weights=weights)
        weighted_avg = (weighted_avg[0] * self.w, weighted_avg[1] * self.h)
        
        # Calculate variance to determine trustworthiness
        if num_frames < 10:  # Not enough data
            return {'centroid': weighted_avg, 'is_trustworthy': False}

        variance_x = np.var(centroids[:, 0])
        variance_y = np.var(centroids[:, 1])

        # Scale variances by width (w) and height (h)
        scaled_variance_x = variance_x * self.w
        scaled_variance_y = variance_y * self.h
        is_trustworthy = variance_x < self.trust_threshold and variance_y < self.trust_threshold*2
        return {'centroid': weighted_avg, 'is_trustworthy': is_trustworthy}
    
    def embed_static(self, image: np.ndarray) -> List[float]:
        """
        Generates an embedding vector from the given image file.
        Raises HTTPException if no face or multiple faces are detected.
        """

        faces = self.face_analysis_model.get(image)
        if len(faces) > 1:
            face = self.getCentralFace(faces, self.w, self.h)
            return face.embedding.tolist()
            
        if len(faces) < 1:
            raise HTTPException(
                status_code=400, detail="No face detected. Please upload an image with a clear face.")

        return faces[0].embedding.tolist()
    def embed(self, image: np.ndarray) -> List[float]:
        """
        Generates an embedding vector from the given image file.
        Raises HTTPException if no face or multiple faces are detected.
        """
        
        faces = self.face_analysis_model.get(image)
        embedding_vectors = []
        if len(faces) < 1:
            raise HTTPException(
                status_code=400, detail="No face detected. Please upload an image with a clear face.")
        elif len(faces) > 1:
            #Lip center
            speaker_location = self.get_speaker_location()
            if (speaker_location['is_trustworthy'] and speaker_location['centroid'] != None):
                lip_x, lip_y = speaker_location['centroid']
                
                matching_faces = []
                for face in faces:
                    x1, y1, x2, y2 = face.bbox
                    if (x1 <= lip_x <= x2) and (y1 <= lip_y <= y2):
                        matching_faces.append(face)
                
                if len(matching_faces) == 1:
                    embedding_vectors = matching_faces[0].embedding.tolist()
                else:
                    return None
            else:
                return None
            
        else:
            embedding_vectors = faces[0].embedding.tolist()
        return embedding_vectors

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

    def getCentralFace(self, faces: List[Face], image_width: int, image_height: int) -> Optional[Face]:
        """
        Get the face closest to the center of the image from a list of faces,
        weighting horizontal (x-axis) distance twice as much as vertical (y-axis) distance.
        
        Args:
            faces: List of detected faces
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            
        Returns:
            Face closest to center (with x-axis prioritized 2:1) or None if empty list
        """
        if not faces:
            return None
            
        image_center = np.array([image_width / 2, image_height / 2])
        closest_face = None
        min_weighted_distance = float('inf')
        
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            face_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            
            # Calculate distances from center
            dx = abs(face_center[0] - image_center[0])
            dy = abs(face_center[1] - image_center[1])
            
            # Apply 2:1 weighting (x distance matters twice as much)
            weighted_distance = 2*dx + dy
            
            if weighted_distance < min_weighted_distance:
                min_weighted_distance = weighted_distance
                closest_face = face
        
        return closest_face
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
