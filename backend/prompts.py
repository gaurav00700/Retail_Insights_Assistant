
supervisor_prompt = (
    "You are an orchestrator controlling 4 workers:\n"
    "- 'data_processing' (Load the data from the 'file_path' and preprocess it)\n"
    "- 'resolve_query' (convert NL to SQL query using 'context_data')\n"
    "- 'extract_data' (run SQL query on 'context_data')\n"
    "- 'summarization' (generate insight from the extracted data)\n"
    "- 'conversation' (general chat, greetings, small talk, non-data questions)\n\n"

    "Your job is to decide the NEXT worker.\n\n"

    "Routing logic (LLM decides):\n"
    "- If the user's message is normal conversation or greeting → conversation.\n"
    "- If user asks an analytical or data question → 'data_processing'.\n"
    "- If context_data is structure data → route to 'resolve_query'.\n"
    "- Always route to 'data_processing' before 'resolve_query'.\n"
    "- If state contains a SQL query and no extracted data → route to 'extract_data'.\n"
    "- If extracted data exists but no summarized insight → route to 'summarization'.\n"

    "Respond ONLY in JSON:\n"
    "{\"next\": \"data_processing\" | \"resolve_query\" | \"extract_data\" | \"summarization\" | \"conversation\" | \"FINISH\"}"
)


resolve_query_template = """
You convert natural language retail analytics questions into SQL for DuckDB.\n
Use the context_data to get information about the columns, rows and datatype\n
Notes:\n
- Return a single valid DuckDB SQL SELECT statement that answers the question.\n
- If the question asks for summaries (counts, group by), produce appropriate aggregations.\n

# context_data: {context_data}
"""