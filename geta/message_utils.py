"""
Helper functions for message processing
"""

from typing import List, Optional
from .models import Message


def build_context_prompt(messages: List[Message]) -> str:
    """Build conversation context prompt from message history"""
    lines = []
    for msg in messages:
        if msg.role == "user":
            lines.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"Assistant called: {tc['function']['name']}")
            elif msg.content:
                lines.append(f"Assistant: {msg.content}")
        elif msg.role == "tool":
            lines.append(f"Tool result: {msg.content}")
    return "\n".join(lines)


def has_tool_result(messages: List[Message]) -> bool:
    """Check if messages contain tool execution results"""
    return any(msg.role == "tool" for msg in messages)


def get_last_user_message(messages: List[Message]) -> Optional[str]:
    """Get the last user message from message history"""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return None
