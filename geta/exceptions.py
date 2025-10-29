"""
Custom exceptions for tool call interception
"""


class ToolCallIntercepted(Exception):
    """Exception raised when a tool call is detected and intercepted"""

    def __init__(self, tool_name: str, tool_arguments: dict):
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments
        super().__init__(f"Tool call: {tool_name}")
