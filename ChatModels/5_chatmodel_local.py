from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

pipe = pipeline(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=pipe)

model = ChatHuggingFace(llm=llm)


result = model.invoke('What is the capital of india?')
print(result)