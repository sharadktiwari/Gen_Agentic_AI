import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

loader = PyPDFLoader("./src/books/Building Machine Learning Systems with Python - Second Edition.pdf")

docs = loader.load()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# To create a vector store from documents
# vector_store = Chroma.from_documents(embedding=embedding_model,
#                                     documents=docs,
#                                     collection_name="pdf_docs",
#                                     persist_directory="./src/vector_store")

# To Add documents to the vector store
# vector_store.add_documents(documents=docs)

# print(vector_store.get(include=['embeddings','documents', 'metadatas']))

vector_store = Chroma(embedding_function=embedding_model,
                    collection_name="pdf_docs",
                    persist_directory="./src/vector_store")

# # response = vector_store.similarity_search(
# response = vector_store.similarity_search_with_score(
#     query='What is Machine Learning?',
#     k=2
# )

response = vector_store.similarity_search_with_score(
    query="",
    filter={"page_label": "10"}
)

print(response)

# for doc, score in response:
#     print(f"Content: {doc.page_content[:100]}")
#     print(f"Metadata: {doc.metadata}")
#     print(f"Score: {score}\n")