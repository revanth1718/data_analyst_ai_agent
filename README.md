
# Data Analyst AI Agent

A production-ready Data Analyst AI Agent that uses LangGraph, FastAPI, and Streamlit to analyze data and generate visualizations.

## Features

- **LangGraph Workflow**: Structured agent workflow (Query Analysis -> Code Generation -> Execution -> Reasoning).
- **FastAPI Backend**: REST API for handling data sessions and agent interactions.
- **Streamlit Frontend**: Interactive web interface for uploading data and chatting with the agent.
- **Code Execution**: safely executes pandas/matplotlib code.
- **NVIDIA AI Endpoints**: Uses Llama-3.1-Nemotron for high-quality reasoning and code generation.

## Project Structure

```
data_analyst_ai_agent/
├── backend/            # FastAPI application
│   ├── app/
│   │   ├── api/        # API endpoints
│   │   ├── core/       # Configuration
│   │   ├── graph/      # LangGraph nodes and workflow
│   │   └── services/   # LLM and Session services
│   └── main.py         # Entry point
├── frontend/           # Streamlit application
│   └── app.py
├── .env.example        # Environment variables example
└── requirements.txt    # Project dependencies
```

## Setup

1.  **Clone the repository**.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables**:
    Copy `.env.example` to `.env` and add your NVIDIA API Key.
    ```bash
    cp .env.example .env
    # Edit .env and set NVIDIA_API_KEY
    ```

## Running the Application

You need to run both the backend and frontend.

### 1. Start the Backend (FastAPI)

Run this in one terminal:
```bash
python -m backend.app.main
```
Or using uvicorn directly (from root):
```bash
uvicorn backend.app.main:app --reload
```
The API will be available at `http://localhost:8000`.

### 2. Start the Frontend (Streamlit)

Run this in a second terminal:
```bash
streamlit run frontend/app.py
```
The frontend will open in your browser (usually `http://localhost:8501`).

## Workflow

1.  Upload a CSV file in the Streamlit sidebar.
2.  Ask questions about the data (e.g., "Analyze the trends", "Plot sales over time").
3.  The agent generates code, executes it, and explains the results.
