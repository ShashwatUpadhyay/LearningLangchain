from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
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
    template="Write a detailed report about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="summarize the following text: {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

report_generation_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    ( lambda x: len(x.split()) > 200, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

#LCEL = LangChain Expression Language == RunnableSequence
final_chain = report_generation_chain | branch_chain

print(final_chain.invoke({"topic": "Russia vs Ukrain"}))