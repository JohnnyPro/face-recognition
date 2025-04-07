from typing import List, Optional
from pydantic import BaseModel, Field
from app.models.schemas import Match
from datetime import datetime

class EmbedResponseV2(BaseModel):
    person_id: str
    embedding_size: int
    status: str
    message: Optional[str] = None
  
class SeedResponse(BaseModel):
    status: str
    already_seeded: bool
    person_id: str
    error: Optional[int] = 0
    message: Optional[str] = None

class MatchTimeTracker:
    person_id: str
    last_seen_time: datetime
    framesPresent: int = 0
    greeted: bool = False
    
    
class IdentifyResponseV2(BaseModel):
   matches: List[Match]
   face_detected: bool
   processed_faces: int
   status: str
   tracked: Optional[List[str]] = []
   new_faces: Optional[List[str]] = []
   lip_center: Optional[List[float]] = []
   error: Optional[str] = None

class SingleFaceResponse(BaseModel):
    match: Optional[Match]
    face_detected: bool
    status: str
    error: Optional[str] = None