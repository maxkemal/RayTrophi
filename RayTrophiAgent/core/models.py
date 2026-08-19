from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class AgentSession:
    agent_id: str
    session_id: str
    
@dataclass
class ToolCall:
    id: str
    name: str
    args: Dict[str, Any]
