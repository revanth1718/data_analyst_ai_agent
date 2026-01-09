
from fastapi import FastAPI
from backend.app.api.endpoints import router as api_router
from backend.app.core.config import get_settings
import uvicorn

settings = get_settings()

app = FastAPI(title="Data Analysis Agent API")

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
