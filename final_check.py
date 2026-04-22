import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# 1. Setup Models (Same as before)
llm = ChatGroq(model_name="llama-3.1-8b-instant")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name="mechanical-ai-portal", embedding=embeddings)

# 2. Testing Logic (Direct Print)
test_question = "What is the maintenance procedure for the impeller?" # Yahan apne PDF ka koi topic likho

print("\n🚀 Connecting to Pinecone...")
# Pinecone se context dhoondna
docs = vectorstore.similarity_search(test_question, k=2)

if not docs:
    print("❌ Error: Pinecone se koi data nahi mila! Check Index name.")
else:
    context = "\n".join([doc.page_content for doc in docs])
    print(f"✅ Context Found from PDF: {context[:300]}...") # Pehle 300 words

    print("\n🤖 AI is thinking...")
    prompt = f"Context: {context}\n\nQuestion: {test_question}\n\nAnswer strictly from the context provided."
    response = llm.invoke(prompt)

    print("\n--- FINAL EVALUATION RESULT ---")
    print(f"QUESTION: {test_question}")
    print(f"ANSWER: {response.content}")