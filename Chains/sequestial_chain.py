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

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following test \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

#LCEL
chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Unimployment in India'})

chain.get_graph().print_ascii()
print(result)