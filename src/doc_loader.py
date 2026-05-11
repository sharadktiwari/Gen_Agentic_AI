from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader, WebBaseLoader, CSVLoader

# Text Loader
# loader = TextLoader("./src/text_example.txt")#, encoding="utf-8")
# docs = loader.load()

# PDF Loaders
# Simple Clean PDFs -> PyPDFLoader
# PDFs with tables/columns -> PDFPlumberLoader
# Scanner/image PDFs -> UnstructuredPDFLoader or AmazonTextractPDFLoader
# Need layout & Image data -> PuMuPDFLoader
# Want best structure exraction -> UnstructuredPDFLoader

# Example -> PDF Loader
# loader = PyPDFLoader("./src/Sharad Resume 4 years.pdf")
# docs = loader.load()

# Directory Loader
# loader = DirectoryLoader(path="./src/books", glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
# docs = loader.load() # Loads all PDFs at once in memory
# docs = loader.lazy_load() # Loads PDFs one at a time when accessed

# WebBase Loader
# loader = WebBaseLoader("https://github.com/campusx-official/langchain-document-loaders/tree/main") # We can pass list of URLs as well
# docs = loader.load()

# CSV Loader
loader = CSVLoader(file_path="./src/Social_Network_Ads.csv", encoding="utf-8")
docs = loader.load()

for doc in docs:
    print(doc.metadata)
    print(doc.page_content)
