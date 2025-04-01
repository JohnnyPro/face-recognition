from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import router as v1_router
from app.api.v2.endpoints import router as v2_router

app = FastAPI(
    title="Face Recognition API",
    description="API for face recognition services with WebSocket support",
    version="2.0",
    redoc_url="/api/redoc"
)

# Configure CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include both API versions
app.include_router(v1_router, prefix="/api/v1", tags=["v1 - REST API"])
app.include_router(v2_router, prefix="/api/v2", tags=["v2 - WebSocket API"])

# Optional: Add lifespan events if needed
@app.on_event("startup")
async def startup_event():
    print("Application starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutting down...")