from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # or another supported model name
    temperature=0.7,
    max_tokens=None,  # use default or set a limit
    timeout=None,
    max_retries=2
)

model2 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # or another supported model name
    temperature=0.7,
    max_tokens=None,  # use default or set a limit
    timeout=None,
    max_retries=2
)

prompt1 = PromptTemplate(
    template="generate a tweet about topic : {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="generate a linkedin about topic : {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, model1, parser),
    'linkedin' : RunnableSequence(prompt2, model2, parser)
})

result = parallel_chain.invoke({'topic':'AI'})

parser2 = StrOutputParser()

print(parser2.invoke(result['tweet'] + result['linkedin']))