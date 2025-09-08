import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.environ.get("NVIDIA_API_KEY")
if not api_key:
    raise ValueError("NVIDIA_API_KEY not found in environment variables")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)
