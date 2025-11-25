import json
import asyncio
from pydantic import BaseModel, Field
from typing import Dict, TypedDict, Literal, Annotated, Optional, Sequence, Type, Union, List, Any
import operator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from backend.backend import chatbot  

app = FastAPI()

# ----------------------------------------------------------------
#  /chat   →   Streaming multi-agent answer
# ----------------------------------------------------------------
# Define the schema
class ChatPayload(BaseModel):
    user_query: str = Field(..., description="User input question")
    file_path: str | None = Field(..., description="Path of the context file")
    thread_id: str = Field(..., description="Unique id foreach thread")

@app.get("/")
async def Welcome():
    """Creating landing page endpoint"""

    return{"message": "Welcome of Chatbot API"}

@app.post("/chat")
async def chat_endpoint(request: ChatPayload):
    """Chat API"""

    # Get Payload
    # body = await request.json()
    # user_query = body.get("query")
    # thread_id = body.get("thread_id", "default")
    # file_path = body.get("file_path")
    user_query = request.user_query
    thread_id = request.thread_id
    file_path = request.file_path

    # Prepare workflow state
    workflow_state = {
        "file_path": file_path,
        "user_query": user_query,
        "messages": [HumanMessage(content=user_query)],
    }

    # LangGraph config
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id}
    }

    # Streaming generator
    async def event_stream():
        streamed_text = ""

        for msg_chunk, metadata in chatbot.stream(
            workflow_state,
            config=config,
            stream_mode="messages"
        ):
            # Get node information
            node = metadata.get("langgraph_node", "")

            # Ignore supervisor, resolver, extractor output
            if node not in ("summarization", "conversation"):
                continue

            streamed_text += msg_chunk.content
            # send incremental chunk
            yield msg_chunk.content

        # send final sentinel
        yield ""

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# To deploy: uvicorn backend.app:app --reload --port 8000
