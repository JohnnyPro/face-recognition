from pydantic import BaseModel
from typing import List, Tuple
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

class IdentifyResponse(BaseModel):
    matches: List[Tuple[Person, float]]  # List of (Person, confidence) tuples
