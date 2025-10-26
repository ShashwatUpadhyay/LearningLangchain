from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('DocumentLoader\\books\\lmao.pdf')   

docs = loader.load()


splitter = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=0,
)


result = splitter.split_documents(docs)

print(result[1].page_content)