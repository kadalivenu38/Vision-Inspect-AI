from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import Base, engine
from app.routes.auth import router as auth_router

app = FastAPI(
    title="VisionInspect AI API",
    version="1.0.0",
    description="Backend API for VisionInspect AI"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth_router)

# Startup Event
@app.on_event("startup")
def startup():
    try:
        # Create database tables
        Base.metadata.create_all(bind=engine)
        print("Database Tables Created Successfully!")

        # Test database connection
        connection = engine.connect()
        print("PostgreSQL Connected Successfully!")
        connection.close()
    except Exception as e:
        print("Database Connection Failed!")
        print(e)


# Root Route
@app.get("/")
async def root():
    return {
        "message": "Welcome to VisionInspect AI Backend"
    }


# Health Check
@app.get("/health")
async def health():
    return {
        "status": "Working Fine"
    }