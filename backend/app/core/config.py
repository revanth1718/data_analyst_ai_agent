
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    NVIDIA_API_KEY: str
    BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    MODEL_NAME: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
    
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding='utf-8',
        extra='ignore'
    )

@lru_cache
def get_settings():
    return Settings()
