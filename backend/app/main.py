from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import engine

app = FastAPI(
    title="VisionInspect AI API",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],  # React Frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# DB Connection
@app.on_event("startup")
def startup():

    try:
        connection = engine.connect()
        print("PostgreSQL Connected Successfully!")
        connection.close()

    except Exception as e:
        print("Database Connection Failed")
        print(e)


# Routes
@app.get("/")
async def root():
    return {
        "message": "Welcome to VisionInspect AI Backend"
    }


@app.get("/health")
async def health():
    return {
        "status": "Working fine"
    }