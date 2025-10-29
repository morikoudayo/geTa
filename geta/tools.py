"""
Tool definitions for agent function calling
"""

from .exceptions import ToolCallIntercepted


def read_file(filepath: str) -> str:
    """
    Use this tool if you need to view the contents of an existing file.

    Args:
        filepath: File path relative to workspace root
    """
    raise ToolCallIntercepted("read_file", {"filepath": filepath})
