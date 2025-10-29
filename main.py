"""
geTa: Gateway Enabling Tool Access

例外ベースのツール呼び出しインターセプトで、
小規模モデルでも正確なFunction Callingを実現。
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal, Iterator
from contextlib import asynccontextmanager
import json
import uuid
import logging
from datetime import datetime
import os

from llama_cpp_agent import LlamaCppAgent, LlamaCppFunctionTool
from llama_cpp_agent.providers import LlamaCppServerProvider
from llama_cpp_agent.providers.llama_cpp_server import LlamaCppSamplingSettings
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings

# ロギング設定（uvicorn形式に完全一致）
class ColoredFormatter(logging.Formatter):
    """uvicornスタイルの色付きログフォーマッター"""
    COLORS = {
        "DEBUG": "\033[36m",      # cyan
        "INFO": "\033[32m",       # green
        "WARNING": "\033[33m",    # yellow
        "ERROR": "\033[31m",      # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(levelname)s:     %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)


# ==================== 例外定義 ====================

class ToolCallIntercepted(Exception):
    """ツール呼び出しを検出した際に発生させる例外"""
    def __init__(self, tool_name: str, tool_arguments: dict):
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments
        super().__init__(f"Tool call: {tool_name}")


# ==================== データモデル ====================

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = True
    tools: Optional[List[Dict[str, Any]]] = None


# ==================== ツール定義 ====================

def read_file(filepath: str) -> str:
    """
    Use this tool if you need to view the contents of an existing file.

    Args:
        filepath: ファイルパス（ワークスペースルートからの相対パス）
    """
    raise ToolCallIntercepted("read_file", {"filepath": filepath})


# ==================== グローバル変数 ====================

agent: Optional[LlamaCppAgent] = None
tools: List[LlamaCppFunctionTool] = []


# ==================== 初期化 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on application startup"""
    server_url = os.getenv("LLAMA_CPP_SERVER_URL", "http://localhost:8080")
    try:
        initialize_agent(server_url)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        logger.info(f"Set LLAMA_CPP_SERVER_URL environment variable (default: {server_url})")
    yield


def initialize_agent(server_url: str):
    """Initialize agent and tools"""
    global agent, tools

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


# ==================== SSEストリーミング生成 ====================

def create_sse_chunk(chat_id: str, model: str, delta: Dict[str, Any], finish_reason: Optional[str] = None) -> str:
    """SSE形式のチャンクを生成"""
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
    """ツール呼び出しをストリーミング形式で返す"""
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    tool_call_id = f"call_{uuid.uuid4().hex[:24]}"

    # 1. role を送信
    yield create_sse_chunk(chat_id, model, {"role": "assistant", "content": None})

    # 2. tool_call 開始（名前とID）
    yield create_sse_chunk(chat_id, model, {
        "tool_calls": [{
            "index": 0,
            "id": tool_call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": ""}
        }]
    })

    # 3. 引数を送信
    yield create_sse_chunk(chat_id, model, {
        "tool_calls": [{
            "index": 0,
            "function": {"arguments": json.dumps(tool_arguments)}
        }]
    })

    # 4. 終了
    yield create_sse_chunk(chat_id, model, {}, "tool_calls")
    yield "data: [DONE]\n\n"


def stream_text(content: str, model: str) -> Iterator[str]:
    """テキストレスポンスをストリーミング形式で返す"""
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"

    # 1. role を送信
    yield create_sse_chunk(chat_id, model, {"role": "assistant", "content": ""})

    # 2. コンテンツを送信
    yield create_sse_chunk(chat_id, model, {"content": content})

    # 3. 終了
    yield create_sse_chunk(chat_id, model, {}, "stop")
    yield "data: [DONE]\n\n"


# ==================== ヘルパー関数 ====================

def build_context_prompt(messages: List[Message]) -> str:
    """メッセージ履歴から会話コンテキストを構築"""
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
    """ツール実行結果が含まれているかチェック"""
    return any(msg.role == "tool" for msg in messages)


def get_last_user_message(messages: List[Message]) -> Optional[str]:
    """最後のユーザーメッセージを取得"""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return None


# ==================== FastAPI アプリケーション ====================

app = FastAPI(title="geTa", version="1.0.0", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI互換のチャット補完エンドポイント"""

    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        # Generate final response if tool results are present
        if has_tool_result(request.messages):
            logger.debug("Generating final response from tool results")

            prompt = build_context_prompt(request.messages)
            prompt += "\n\nBased on the tool results above, provide a helpful response."

            response = agent.get_chat_response(
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
            output_settings = LlmStructuredOutputSettings.from_llama_cpp_function_tools(tools)
            response = agent.get_chat_response(user_msg, structured_output_settings=output_settings)

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


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn

    logger.info("geTa: Gateway Enabling Tool Access v1.0")
    logger.info("Architecture: Continue → geTa → llama-cpp-agent → llama.cpp server")
    logger.info("Continue config: API Base=http://localhost:8000/v1")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")