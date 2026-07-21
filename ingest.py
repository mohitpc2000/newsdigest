from langchain_community.document_loaders import RSSFeedLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = RSSFeedLoader(urls=["https://feeds.bbci.co.uk/news/technology/rss.xml"])
documents = loader.load()
print(f"Loaded {len(documents)} documents from the RSS feed.")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")
print(f"chunks[0]: {chunks[0]}")