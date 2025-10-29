"""
Data models for OpenAI-compatible API
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal


class Message(BaseModel):
    """OpenAI-compatible message model"""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = True
    tools: Optional[List[Dict[str, Any]]] = None
