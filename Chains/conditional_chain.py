from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableBranch, RunnableLambda
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # or another supported model name
    temperature=0.7,
    max_tokens=None,  # use default or set a limit
    timeout=None,
    max_retries=2
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='sentiment of the feedback')
    language: str
    intent: str

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative  \n {text} \n {format_instruction}',
    input_variables=['text'],
    partial_variables={'format_instruction': parser2.get_format_instructions()},
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropirate response to this positive feedback in 20-50 words  \n {feedback}',
    input_variables=['feedback'],
    
)
prompt3 = PromptTemplate(
    template='Write an appropirate response to this negative feedback in 20-50 words \n {feedback}',
    input_variables=['feedback'],
    
)

branch_chain = RunnableBranch(
    # (condition1 , chain1),
    # (condition2 , chain2),
    # (default , chain),
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain
chain.get_graph().print_ascii()
result = chain.invoke({'text':'This is terriable smartphone!'})
print(result)