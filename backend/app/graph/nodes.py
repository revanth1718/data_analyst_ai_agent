
from typing import Any, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from backend.app.graph.state import AgentState
from backend.app.services.llm import get_llm
from backend.app.services.session_manager import session_manager
import pandas as pd
import matplotlib.pyplot as plt
import io
import traceback

llm = get_llm()

def query_understanding_node(state: AgentState) -> Dict:
    query = state["query"]
    messages = [
        SystemMessage(content="detailed thinking off. You are an assistant that determines if a query is requesting a data visualization. Respond with only 'true' if the query is asking for a plot, chart, graph, or any visual representation of data. Otherwise, respond with 'false'."),
        HumanMessage(content=query)
    ]
    response = llm.invoke(messages)
    intent = response.content.strip().lower() == "true"
    return {"should_plot": intent}

def extract_code(text: str) -> str:
    start = text.find("```python")
    if start == -1: return text
    start += len("```python")
    end = text.find("```", start)
    return text[start:end].strip() if end != -1 else text[start:].strip()

def code_generation_node(state: AgentState) -> Dict:
    cols = state["columns"]
    query = state["query"]
    should_plot = state["should_plot"]
    
    if should_plot:
        prompt_text = f"""
        Given DataFrame `df` with columns: {', '.join(cols)}
        Write Python code using pandas **and matplotlib** (as plt) to answer:
        "{query}"

        Rules
        -----
        1. Use pandas for data manipulation and matplotlib.pyplot (as plt) for plotting.
        2. Assign the final result (DataFrame, Series, scalar *or* matplotlib Figure) to a variable named `result`.
        3. Create only ONE relevant plot. Set `figsize=(10,6)`, add title/labels.
        4. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
        """
    else:
        prompt_text = f"""
        Given DataFrame `df` with columns: {', '.join(cols)}
        Write Python code (pandas **only**, no plotting) to answer:
        "{query}"

        Rules
        -----
        1. Use pandas operations on `df` only.
        2. Assign the final result to `result`.
        3. Wrap the snippet in a single ```python code fence (no extra prose).
        """
        
    messages = [
        SystemMessage(content="detailed thinking off. You are a Python data-analysis expert who writes clean, efficient code. Solve the given problem with optimal pandas operations. Be concise and focused. Your response must contain ONLY a properly-closed ```python code block with no explanations before or after. Ensure your solution is correct, handles edge cases, and follows best practices for data analysis."),
        HumanMessage(content=prompt_text)
    ]
    
    response = llm.invoke(messages)
    code = extract_code(response.content)
    return {"code": code}

def execution_node(state: AgentState) -> Dict:
    code = state["code"]
    session_id = state["session_id"]
    should_plot = state["should_plot"]
    
    df = session_manager.get_df(session_id)
    if df is None:
        return {"code_result": "Error: Session expired or invalid."}
        
    env = {"pd": pd, "df": df}
    if should_plot:
        plt.clf() # Clear previous plots
        plt.rcParams["figure.dpi"] = 100
        env["plt"] = plt
        env["io"] = io
        
    try:
        exec(code, {}, env)
        result = env.get("result", None)
        return {"code_result": result}
    except Exception:
        return {"code_result": f"Error executing code: {traceback.format_exc()}"}

def reasoning_node(state: AgentState) -> Dict:
    query = state["query"]
    result = state["code_result"]
    
    is_error = isinstance(result, str) and result.startswith("Error")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))
    
    desc = ""
    if is_error:
        desc = result
    elif is_plot:
        desc = "[Plot Generated]"
    else:
        desc = str(result)[:500]
        
    if is_plot:
        prompt = f'''The user asked: "{query}". A plot was generated. Explain in 2–3 concise sentences what the chart shows.'''
    else:
        prompt = f'''The user asked: "{query}". The result is: {desc}. Explain in 2–3 concise sentences what this tells about the data.'''
        
    messages = [
        SystemMessage(content="detailed thinking on. You are an insightful data analyst."),
        HumanMessage(content=prompt)
    ]
    
    # We aren't streaming here for simplicity in this node, but the frontend could request stream
    response = llm.invoke(messages)
    
    # Simple cleanup of <think> tags if any (the model might produce them)
    content = response.content
    final_reasoning = content.split("</think>")[-1].strip() if "</think>" in content else content
    
    return {"reasoning": final_reasoning}
