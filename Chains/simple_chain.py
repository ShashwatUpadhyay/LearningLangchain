from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # or another supported model name
    temperature=0.7,
    max_tokens=None,  # use default or set a limit
    timeout=None,
    max_retries=2
)

prompt = PromptTemplate(
    template='Generate 5 interesting fact about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

#LCEL
chain = prompt | model | parser

result = chain.invoke({'topic':'India'})

chain.get_graph().print_ascii()
print(result)