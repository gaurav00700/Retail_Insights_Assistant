import os, sys
parent_dir = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(0, parent_dir)  # add repo entrypoint to python path
import uuid
import streamlit as st
import requests
from langchain_core.messages import HumanMessage, AIMessage
try: 
    from backend.backend import chatbot 
except: 
    pass

# ============================================================
# Environment variables
# ============================================================
BACKEND_API = os.getenv("BACKEND_URL", "http://localhost:8000")

# ============================================================
# Utilities
# ============================================================

def generate_thread_id():
    """generating unique id"""
    return str(uuid.uuid4())

def reset_chat(clear_file: bool = True):
    """Reset chatbot + optionally clear uploaded file widget."""

    if clear_file:
        st.session_state["file_path"] = None

    # reset thread + messages
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []

    # Reset Session state
    st.session_state["chatbot_state"] = {
        "file_path": st.session_state.get("file_path"),
        "messages": [],
        "context_data": None,
        "resolved_query": "",
        "extracted_answer": "",
        "summarized_answer": "",
        "resolve_history": [],
        "extract_history": [],
        "summarized_history": [],
        "worker_history": [],
    }


# ============================================================
# Session Initialization
# ============================================================
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "file_path" not in st.session_state:
    st.session_state["file_path"] = None

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chatbot_state" not in st.session_state:
    reset_chat()

# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("Settings")
st.sidebar.markdown(f"**Thread ID:** `{st.session_state['thread_id']}`")
st.sidebar.markdown(f"**Current file path:**`{st.session_state['file_path']}`")

st.sidebar.subheader("📂 Dataset Source")

# Manual text input to change CSV path
# new_path = st.sidebar.text_input(
#     "Enter file path:",
#     value=st.session_state["file_path"]
# )

# Manual path update button
# if st.sidebar.button("Update file Path", use_container_width=True):
#     st.session_state["file_path"] = new_path
#     reset_chat()
#     st.sidebar.success("File path updated!")
#     st.rerun()

# Upload files and handle it
uploaded_file = st.sidebar.file_uploader(
    "Upload a file",
    type=["csv", "xls", "json", "txt"],
    accept_multiple_files=False,
    key=st.session_state["file_uploader_key"]
)

if uploaded_file:
    filename = uploaded_file.name   # file name
    temp_path = f"data/temp/{filename}" # dire name
    os.makedirs(os.path.dirname(temp_path), exist_ok=True) # Create directory 

    # Process only if new file uploaded
    if st.session_state.get("file_path") != temp_path:

        # Save file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state["file_path"] = temp_path
        st.sidebar.success(f"File saved as: `{temp_path}`")

        # Reset chat but NOT the uploader
        # reset_chat(clear_file=False)

        st.rerun()

# New chat button
if st.sidebar.button("🆕 New Chat", use_container_width=True):
    # Increment key to reset the file_uploader widget
    st.session_state["file_uploader_key"] += 1

    reset_chat(clear_file=True)
    st.rerun()

# ============================================================
# Main UI
# ============================================================
st.title("Multi-Agent Chatbot")

# Display message history
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================================
# Input Box
# ============================================================
user_input = st.chat_input("Ask a question…")

if user_input:
    # Save user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get workflow state
    chatbot_state = st.session_state["chatbot_state"]
    chatbot_state["user_query"] = user_input
    chatbot_state["file_path"] = st.session_state["file_path"]

    # Streaming config
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
        "metadata": {"thread_id": st.session_state["thread_id"]},
    }

    # Streaming block
    with st.chat_message("assistant"):
        stream_box = st.empty()
        stream_box.markdown("⏳ Thinking…")

        streamed_text = ""
        first_token = False

        # Option 1: Generating response through API call 
        response = requests.post(
            url= f"{BACKEND_API}/chat",
            json= {
                "user_query": user_input,
                "thread_id": st.session_state["thread_id"],
                "file_path": st.session_state["file_path"]
                },
            stream= True
            )
        
        for chunk in response.iter_lines():
            if chunk:
                decoded = chunk.decode("utf-8")

                # break on end signal
                if decoded == "[[END]]":
                    break

                if not first_token:
                    stream_box.empty()
                    first_token = True

                streamed_text += decoded
                stream_box.markdown(streamed_text)

        # Option 2: Generating output by invoking the graph
        # for msg_chunk, metadata in chatbot.stream(
        #     {
        #         **chatbot_state,
        #         "messages": [HumanMessage(content=user_input)]
        #     },
        #     config=CONFIG,
        #     stream_mode="messages"
        # ):
        #     node_name = metadata.get("langgraph_node", "")

        #     # Only show final assistant messages
        #     if node_name not in ("summarization", "conversation"):
        #         continue

        #     # Remove the spinner on first output
        #     if not first_token :
        #         stream_box.empty()
        #         first_token = True

        #     streamed_text += msg_chunk.content
        #     stream_box.markdown(streamed_text)

        final_answer = streamed_text.strip() or "No answer."

        # Add assistant message
        st.session_state["message_history"].append(
            {"role": "assistant", "content": final_answer}
        )

    # Update session state through direct graph state
    # final_state = chatbot.invoke(
    #     {**chatbot_state, "messages": [HumanMessage(content=user_input)]},
    #     config=CONFIG
    # )
    # st.session_state["chatbot_state"] = final_state

st.divider()
st.caption("💡 Powered by LangGraph + Multi-Agent Reasoning + DuckDB SQL")

# To Run: streamlit run frontend/app.py