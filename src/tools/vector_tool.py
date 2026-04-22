from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import os

@tool
def mechanical_pdf_search(query: str):
    """Useful when you need to answer technical questions about mechanical parts, 
    maintenance, or engineering procedures from the uploaded manuals."""
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index_name="mechanical-ai-portal", embedding=embeddings)
    
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    return context