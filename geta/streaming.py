"""
Server-Sent Events (SSE) streaming response generation
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Iterator


def create_sse_chunk(
    chat_id: str,
    model: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None
) -> str:
    """Generate a single SSE chunk in OpenAI format"""
    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    }
    return f"data: {json.dumps(chunk)}\n\n"


def stream_tool_call(tool_name: str, tool_arguments: Dict[str, Any], model: str) -> Iterator[str]:
    """Stream a tool call response in OpenAI format"""
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    tool_call_id = f"call_{uuid.uuid4().hex[:24]}"

    # 1. Send role
    yield create_sse_chunk(chat_id, model, {"role": "assistant", "content": None})

    # 2. Start tool_call with name and ID
    yield create_sse_chunk(chat_id, model, {
        "tool_calls": [{
            "index": 0,
            "id": tool_call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": ""}
        }]
    })

    # 3. Send arguments
    yield create_sse_chunk(chat_id, model, {
        "tool_calls": [{
            "index": 0,
            "function": {"arguments": json.dumps(tool_arguments)}
        }]
    })

    # 4. Finish
    yield create_sse_chunk(chat_id, model, {}, "tool_calls")
    yield "data: [DONE]\n\n"


def stream_text(content: str, model: str) -> Iterator[str]:
    """Stream a text response in OpenAI format"""
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"

    # 1. Send role
    yield create_sse_chunk(chat_id, model, {"role": "assistant", "content": ""})

    # 2. Send content
    yield create_sse_chunk(chat_id, model, {"content": content})

    # 3. Finish
    yield create_sse_chunk(chat_id, model, {}, "stop")
    yield "data: [DONE]\n\n"
