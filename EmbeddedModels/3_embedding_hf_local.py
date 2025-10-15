from langchain_huggingface import HuggingFaceEmbeddings



embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # model_kwargs={"device": "cpu"},
)

text = 'Delhi is the capital of india'

documents = [
    'Delhi is the capital of india',
    'kolkata is the capital of west bengal',
    'mumbai is the capital of maharashtra',
    'chennai is the capital of tamil nadu',
    'bangalore is the capital of karnataka',
]

vector = embedding.embed_documents(documents)

print(str(vector))