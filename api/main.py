from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import data, ai
import uvicorn
import config

app = FastAPI(
    title=config.APP_TITLE,
    description="Backend API for Data Analysis Assistant",
    version="1.0.0"
)

# CORS setup for allowing frontend to access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allow all. In prod, lock this down to the frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix="/api/data", tags=["Data"])
app.include_router(ai.router, prefix="/api/ai", tags=["AI"])

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Data Analysis Assistant API"}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
