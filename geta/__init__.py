"""
llama.geta: Gateway Enabling Tool Access

Exception-based tool call interception for accurate function calling with small models.
"""

__version__ = "1.0.0"
__author__ = "llama.geta Contributors"

from .exceptions import ToolCallIntercepted
from .models import Message, ChatCompletionRequest

__all__ = [
    "ToolCallIntercepted",
    "Message",
    "ChatCompletionRequest",
    "__version__",
]
