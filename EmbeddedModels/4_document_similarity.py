from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # model_kwargs={"device": "cpu"},
)

players = [
  "Virat Kohli is known for his aggressive batting style and consistency across all formats of the game.",
  "Rohit Sharma holds the record for the highest individual score in ODI cricket — 264 runs.",
  "Jasprit Bumrah is one of the world’s best fast bowlers, famous for his deadly yorkers and unique bowling action.",
  "Ravindra Jadeja is a dynamic all-rounder, excelling in both spin bowling and explosive lower-order batting.",
  "KL Rahul is a technically gifted batsman who can adapt to multiple roles, from opener to middle-order anchor."
]


querry = 'tell me about bhumra'

doc_embeddings = embedding.embed_documents(players)
querry_embedding = embedding.embed_query(querry)

scores = cosine_similarity([querry_embedding], doc_embeddings)[0]

index, score = sorted((list(enumerate(scores))), key=lambda x:x[1])[-1]

print(querry)
print(players[index])
print("similarity score is: ", score)