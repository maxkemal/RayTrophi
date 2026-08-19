import uuid
from core.models import AgentSession

class SessionManager:
    def __init__(self):
        self.active_session = AgentSession(
            agent_id=f"agent_{uuid.uuid4().hex[:6]}",
            session_id=f"session_{uuid.uuid4().hex[:6]}"
        )
        
    def get_session(self) -> AgentSession:
        return self.active_session
