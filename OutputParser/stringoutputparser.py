from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    # max_tokens=None, 
    # timeout=None,
    # max_retries=2
)

# 1st prompt -> detailed report

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']   
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='write a 5 line summary on the follow text. /n {text}',
    input_variables=['text']   
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'Black Hole'})
print(result)