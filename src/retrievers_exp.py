import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from dotenv import load_dotenv
load_dotenv()
from langchain_community.retrievers import WikipediaRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

# Wikipedia Retriever
# retriever = WikipediaRetriever(top_k_results=2, lang="en")
# query = "Geopolitical history of India"

# Chroma Retriever
loader = PyPDFLoader("./data/books/Building Machine Learning Systems with Python - Second Edition.pdf")
docs = loader.load()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vector_store = Chroma.from_documents(embedding=embedding_model,
#                                     documents=docs,
#                                     collection_name="retriever_docs",
#                                     persist_directory="./vector_store")
vector_store = Chroma(embedding_function=embedding_model,
                    collection_name="retriever_docs",
                    persist_directory="./vector_store")

# Normal Search
# retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# MMR Search
# retriever = vector_store.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 2, "lambda_mult": 0.5})

# Multi Query Retriever
# llm = HuggingFaceEndpoint(
#     repo_id= "Qwen/Qwen2.5-7B-Instruct",
#     task="text-generation",
#     temperature=0.5
#     )

# model = ChatHuggingFace(llm=llm)

# retriever = MultiQueryRetriever.from_llm(
#     retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
#     # search_type="mmr",
#     llm = model
# )

# Contextual Compression Retriever
llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.5
    )

model = ChatHuggingFace(llm=llm)
compressor = LLMChainExtractor.from_llm(model)

retriever = ContextualCompressionRetriever(
    base_retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    base_compressor=compressor
)

query = "What is Machine Learning?"

docs = retriever.invoke(query)

for doc in docs:
    print(doc.page_content[:200])
