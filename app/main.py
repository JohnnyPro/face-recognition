from fastapi import FastAPI
from .api import router

app = FastAPI()
app.include_router(router, prefix="/api/v1")