from langchain_community.document_loaders import RSSFeedLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
loader = RSSFeedLoader(urls=["https://feeds.bbci.co.uk/news/technology/rss.xml"])
documents = loader.load()
print(f"Loaded {len(documents)} documents from the RSS feed.")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")
print(f"chunks[0]: {chunks[0]}")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
chunks = filter_complex_metadata(chunks)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
print("Documents embedded and stored in ChromaDB at './chroma_db'.")
print(f"Stored {vectorstore._collection.count()} chunks in Chroma.")