from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-3.5-turbo',temperature=0, max_completion_tokens=10)

model.invoke("what is the capital of India") 