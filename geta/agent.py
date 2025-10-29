"""
Agent initialization and state management
"""

import os
import logging
from typing import Optional, List

from llama_cpp_agent import LlamaCppAgent, LlamaCppFunctionTool
from llama_cpp_agent.providers import LlamaCppServerProvider
from llama_cpp_agent import MessagesFormatterType
from .tools import read_file

logger = logging.getLogger(__name__)

# Global state
agent: Optional[LlamaCppAgent] = None
tools: List[LlamaCppFunctionTool] = []


def initialize(server_url: Optional[str] = None) -> None:
    """
    Initialize the agent and tools synchronously.
    Must be called before starting the server.

    Args:
        server_url: llama.cpp server URL (defaults to LLAMA_CPP_SERVER_URL env var or http://localhost:8080)
    """
    global agent, tools

    if server_url is None:
        server_url = os.getenv("LLAMA_CPP_SERVER_URL", "http://localhost:8080")

    logger.info(f"llama.cpp server: {server_url}")

    provider = LlamaCppServerProvider(server_address=server_url)
    agent = LlamaCppAgent(
        provider,
        system_prompt="You are a helpful coding assistant. Use the available tools when needed.",
        predefined_messages_formatter_type=MessagesFormatterType.CHATML,
        debug_output=False
    )
    tools = [LlamaCppFunctionTool(read_file)]

    tool_names = [t.model.model_json_schema()["title"] for t in tools]
    logger.info(f"Initialized with tools: {tool_names}")


def is_initialized() -> bool:
    """Check if agent is initialized"""
    return agent is not None
