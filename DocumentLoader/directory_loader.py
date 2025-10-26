from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='DocumentLoader\\books',
    glob='*.pdf',
    loader_cls=PyPDFLoader,
    show_progress=True
)

# docs = loader.load()
# docs = loader.lazy_load()

for load in loader.lazy_load():
    # print(load)
    pass

# print(docs[100].page_content)
# print(docs[100].metadata)