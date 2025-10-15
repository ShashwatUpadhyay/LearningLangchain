from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    'Delhi is the capital of india',
    'kolkata is the capital of west bengal',
    'mumbai is the capital of maharashtra',
    'chennai is the capital of tamil nadu',
    'bangalore is the capital of karnataka',
    'hyderabad is the capital of telangana',
    'pune is the capital of maharashtra',
    'jaipur is the capital of rajasthan',
    'lucknow is the capital of uttar pradesh',
    'kanpur is the capital of uttar pradesh',
    'ahmedabad is the capital of gujarat',
    'surat is the capital of gujarat',
    'kochi is the capital of kerala',
    'trivandrum is the capital of kerala',
    'bhopal is the capital of madhya pradesh',
    'indore is the capital of madhya pradesh',
    'jaipur is the capital of rajasthan',
    'chennai is the capital of tamil nadu',
    'coimbatore is the capital of tamil nadu',
    'madras is the capital of tamil nadu',
    'madras is the capital of tamil nadu',
    'madras is the capital of tamil nadu'
]

result = embedding.aembed_documents(documents)

print(str(result))