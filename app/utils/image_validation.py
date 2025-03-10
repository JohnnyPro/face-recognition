from fastapi import HTTPException
from typing import List

# List of allowed image MIME types
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]

def validate_image_file(file) -> None:
    """
    Validates that the uploaded file is an image.
    Raises HTTPException if the file is not an image.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types are: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )