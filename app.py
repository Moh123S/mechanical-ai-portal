import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval_qa.base import RetrievalQA

load_dotenv()

# 1. Setup LLM & Embeddings
llm = ChatGroq(
    temperature=0.1, 
    model_name="llama-3.1-8b-instant", 
    groq_api_key=os.getenv("GROQ_API_KEY")
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Connect to Pinecone
index_name = "mechanical-ai-portal"
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# 3. Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

def ask_ai():
    print("\n🤖 Mechanical AI: Haan Mohit bhai, pucho kya puchna hai? (exit likho band karne ke liye)")
    while True:
        user_input = input("👨‍🔧 Mohit: ")
        if user_input.lower() == 'exit':
            break
        
        try:
            response = qa_chain.invoke(user_input)
            print(f"\n🤖 AI: {response['result']}\n")
        except Exception as e:
            print(f"Oops, error aa gaya: {e}")

if __name__ == "__main__":
    ask_ai()