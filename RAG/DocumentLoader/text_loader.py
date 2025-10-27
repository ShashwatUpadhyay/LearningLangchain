from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # or another supported model name
    temperature=0.7,
    max_tokens=None,  # use default or set a limit
    timeout=None,
    max_retries=2
)

loader = TextLoader('DocumentLoader\poem.txt', encoding='utf-8')

prompt = PromptTemplate(
    template='Generate a summary from the following poem:  \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

doc = loader.load()

# print(doc[0].page_content)
# print(doc[0].metadata)
# print(doc[0].__doc__)

chain = prompt | model | parser

result = chain.invoke({'poem':doc[0].page_content})

print(result)