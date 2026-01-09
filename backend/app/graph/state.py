
from typing import TypedDict, Annotated, List, Any, Union, Dict
from langgraph.graph.message import add_messages
import pandas as pd
import matplotlib.pyplot as plt

class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    query: str
    df_head: str  # String representation of head for the LLM
    columns: List[str]
    code: str
    code_result: Any
    reasoning: str
    should_plot: bool
    session_id: str
