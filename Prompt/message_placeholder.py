from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages  import HumanMessage

# chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# load chat historyy
chat_history = []

with open('Prompt/chat_history.txt','r') as f:
    chat_history.extend(f.readlines())
    
print(chat_history)

#chat prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query' : 'Where is my refund'})

print(prompt)