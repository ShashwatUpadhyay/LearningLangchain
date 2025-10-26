from langchain_community.document_loaders import WebBaseLoader
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
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)
parser = StrOutputParser()

url = 'https://www.amazon.in/ZEBRONICS-GT740-4GD3-Graphics-Powered-Multiple/dp/B0DGT7B4JX?ref_=Oct_d_obs_d_1375354031_0'
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt1 | model | parser


result = chain.invoke({'question':'list me all the specification of that grafic card.','text':docs[0].page_content})

print(result)