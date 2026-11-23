from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# Returns Prompt Template
def get_prompt():
    prompt = ChatPromptTemplate([
    ('system','You are A Helpfull AI Assistant, Answer from this only - {retrieved_context} \n If Not know Say "I am Sorry"'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])
    return prompt