import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Models load karo
llm = ChatGroq(model_name="llama-3.1-8b-instant")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name="mechanical-ai-portal", embedding=embeddings)

def check_accuracy(question):
    # 1. Pinecone se context lao
    docs = vectorstore.similarity_search(question, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 2. AI se jawab mango
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer only from context."
    response = llm.invoke(prompt)
    
    print(f"\n🔍 Question: {question}")
    print(f"📄 Context Found (PDF se): {context[:200]}...") # Shuruat ka thoda sa context
    print(f"🤖 AI Answer: {response.content}")

if __name__ == "__main__":
    q = input("Ek tough sawal pucho apne PDF se: ")
    check_accuracy(q)