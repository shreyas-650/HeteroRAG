from langchain_community.chat_message_histories import ChatMessageHistory

chat_history = {}

# Returns Chat History
def get_history(session_id: str):
    if session_id not in chat_history:
        chat_history[session_id]=ChatMessageHistory()
    return chat_history[session_id]