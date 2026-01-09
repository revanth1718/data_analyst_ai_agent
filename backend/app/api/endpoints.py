
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
import io
from starlette.responses import StreamingResponse

from backend.app.services.session_manager import session_manager
from backend.app.graph.workflow import app_graph

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    query: str

class ChatResponse(BaseModel):
    response: str
    code: str
    plot_id: Optional[int] = None

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        session_id = session_manager.create_session(df)
        
        # Initial analysis (optional, can be moved to a graph node)
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        return {
            "session_id": session_id,
            "filename": file.filename,
            "columns": df.columns.tolist(),
            "info": info_str
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    df = session_manager.get_df(request.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
        
    initial_state = {
        "messages": [],
        "query": request.query,
        "df_head": df.head().to_string(),
        "columns": df.columns.tolist(),
        "session_id": request.session_id,
        "should_plot": False
    }
    
    result = app_graph.invoke(initial_state)
    
    code_result = result["code_result"]
    plot_id = None
    
    # Check if result is a plot
    if isinstance(code_result, (plt.Figure, plt.Axes)):
        fig = code_result.figure if isinstance(code_result, plt.Axes) else code_result
        plot_id = session_manager.add_plot(request.session_id, fig)
    
    return ChatResponse(
        response=result["reasoning"],
        code=result["code"],
        plot_id=plot_id
    )

@router.get("/plots/{session_id}/{plot_id}")
async def get_plot(session_id: str, plot_id: int):
    fig = session_manager.get_plot(session_id, plot_id)
    if not fig:
        raise HTTPException(status_code=404, detail="Plot not found")
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
