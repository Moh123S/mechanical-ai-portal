import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def run_ingestion():
    # 1. Load PDFs from data folder
    loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Push to Pinecone
    index_name = "mechanical-ai-portal"
    PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)
    print("Ingestion Successful! Data is in Pinecone.")

if __name__ == "__main__":
    run_ingestion()