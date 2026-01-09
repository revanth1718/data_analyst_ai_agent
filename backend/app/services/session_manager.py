
import pandas as pd
from typing import Dict, Optional
import uuid

class SessionManager:
    _instance = None
    _sessions: Dict[str, pd.DataFrame] = {}
    _plots: Dict[str, list] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance

    def create_session(self, df: pd.DataFrame) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = df
        self._plots[session_id] = []
        return session_id

    def get_df(self, session_id: str) -> Optional[pd.DataFrame]:
        return self._sessions.get(session_id)
    
    def add_plot(self, session_id: str, figure):
        if session_id not in self._plots:
            self._plots[session_id] = []
        self._plots[session_id].append(figure)
        return len(self._plots[session_id]) - 1

    def get_plot(self, session_id: str, index: int):
        if session_id in self._plots and 0 <= index < len(self._plots[session_id]):
            return self._plots[session_id][index]
        return None

session_manager = SessionManager()
