from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="DocumentLoader\docs\mock_questions.csv")

docs = loader.load()

print(len(docs))
print(docs[0])