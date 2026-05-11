from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_text_splitters import Language

# Text Loader
loader = TextLoader("./data/text_example.txt")#, encoding="utf-8")
docs = loader.load()
text = docs[0].page_content

# Character Text Splitter
# splitter = CharacterTextSplitter(
#     chunk_size=200,
#     chunk_overlap=0,
#     separator=''
# )

# Recursive Character Text Splitter
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=200,
#     chunk_overlap=0
# )

# Recursive Character Text Splitter For Markdown/Codes
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0
)

result = splitter.split_documents(docs)

print(result[1].page_content)

