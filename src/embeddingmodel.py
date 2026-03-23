from langchain_huggingface import HuggingFaceEmbeddings 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


embd_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs=dict(
                                            device="cpu"
                                        ))

docs = [
    "Lonely, I am so lonely, I have nobody to love",
    "You just want attention, you don't want my heart",
    "I will always be around wherever life takes you, you know I'll follow you"
]

doc_vectors = embd_model.embed_documents(docs)

query = "get me Follow You song ?"
query_vector = embd_model.embed_query(query)

similarities = cosine_similarity([query_vector], doc_vectors)[0]

print(docs[np.argmax(similarities)])
