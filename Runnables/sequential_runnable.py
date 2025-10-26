from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
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
    template="Tell me a short joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke: {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt1, model, parser)

print(chain.invoke({"topic": "ice cream"}))