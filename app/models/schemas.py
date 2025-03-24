from pydantic import BaseModel
from typing import List, Tuple, Optional
from fastapi import UploadFile, File


class Person(BaseModel):
    id: int
    name: str

class EmbedRequest(BaseModel):
    name: str
    image: UploadFile  = File(...)

class EmbedResponse(BaseModel):
    person: Person

class IdentifyRequest(BaseModel):
    image: UploadFile = File(...)

class Match(BaseModel):
    person_id: str
    confidence: Optional[float] = None
    bbox: Optional[List[float]] = None

class IdentifyResponse(BaseModel):
    matches: List[Match]
    face_detected: bool

    
class Face(BaseModel):
    bbox: List[float]
    embeddings: List[float]
