import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.messages import HumanMessage
from backend.backend import chatbot  

app = FastAPI()

# ----------------------------------------------------------------
#  /chat   →   Streaming multi-agent answer
# ----------------------------------------------------------------
@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    user_query = body.get("query")
    thread_id = body.get("thread_id", "default")
    file_path = body.get("file_path")

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
