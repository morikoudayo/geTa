"""
FastAPI application and endpoints
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from llama_cpp_agent.providers.llama_cpp_server import LlamaCppSamplingSettings
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings

from .models import ChatCompletionRequest
from .exceptions import ToolCallIntercepted
from .streaming import stream_tool_call, stream_text
from .message_utils import build_context_prompt, has_tool_result, get_last_user_message
from . import agent as agent_module

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    Note: agent must be initialized before calling this function.
    """
    if not agent_module.is_initialized():
        raise RuntimeError("Agent must be initialized before creating the app. Call agent.initialize() first.")

    app = FastAPI(title="llama.geta", version="1.0.0")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """OpenAI-compatible chat completion endpoint"""

        if agent_module.agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        try:
            # Generate final response if tool results are present
            if has_tool_result(request.messages):
                logger.debug("Generating final response from tool results")

                prompt = build_context_prompt(request.messages)
                prompt += "\n\nBased on the tool results above, provide a helpful response."

                response = agent_module.agent.get_chat_response(
                    prompt,
                    llm_sampling_settings=LlamaCppSamplingSettings(
                        temperature=request.temperature
                    )
                )

                return StreamingResponse(
                    stream_text(response, request.model),
                    media_type="text/event-stream"
                )

            # Initial request: determine if tool call is needed
            user_msg = get_last_user_message(request.messages)
            if not user_msg:
                raise HTTPException(status_code=400, detail="No user message found")

            logger.debug(f"User message: {user_msg[:50]}...")

            try:
                # Exception will be raised if tool call is needed
                output_settings = LlmStructuredOutputSettings.from_llama_cpp_function_tools(agent_module.tools)
                response = agent_module.agent.get_chat_response(user_msg, structured_output_settings=output_settings)

                return StreamingResponse(
                    stream_text(response, request.model),
                    media_type="text/event-stream"
                )

            except ToolCallIntercepted as e:
                logger.info(f"Tool call: {e.tool_name}({e.tool_arguments})")

                return StreamingResponse(
                    stream_tool_call(e.tool_name, e.tool_arguments, request.model),
                    media_type="text/event-stream"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app
