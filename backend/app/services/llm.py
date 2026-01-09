
from langchain_openai import ChatOpenAI
from backend.app.core.config import get_settings

settings = get_settings()

def get_llm(temperature: float = 0.2):
    return ChatOpenAI(
        base_url=settings.BASE_URL,
        api_key=settings.NVIDIA_API_KEY,
        model=settings.MODEL_NAME,
        temperature=temperature,
        max_tokens=1024
    )
