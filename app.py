import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 1. Setup LLM & Embeddings
llm = ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Connect to Pinecone
index_name = "mechanical-ai-portal"
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
retriever = vectorstore.as_retriever()

# 3. Contextualize Question (Query Rewriting Logic)
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Ye retriever ab purani baatein yaad rakhega
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# 4. Answer Question Prompt
system_prompt = (
    "You are a Senior Mechanical Engineer Mentor. "
    "Use the following pieces of retrieved context to answer the question. "
    "Strictly follow these rules:\n"
    "1. Answer in a mix of Hindi and English (Hinglish) so it's easy to understand.\n"
    "2. Keep technical terms like 'Impeller', 'Casing', 'Cavitation', 'Alignment' in English only. "
    "Do NOT translate them into difficult Hindi words.\n"
    "3. Explain like you are talking to a colleague on the shop floor.\n"
    "4. If you don't know the answer based on context, just say 'Bhai, iska manual mein zikr nahi hai'.\n\n"
    "Context:\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 5. Chat History list
chat_history = []

def ask_ai():
    print("\n🤖 Mechanical AI (Advanced): Haan Mohit bhai, pucho! (exit likho band karne ke liye)")
    while True:
        user_input = input("👨‍🔧 Mohit: ")
        if user_input.lower() == 'exit': break
        
        # Calling the Advanced Chain
        response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
        
        print(f"\n🤖 AI: {response['answer']}\n")
        
        # History Update karna zaroori hai
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response["answer"]),
        ])

if __name__ == "__main__":
    ask_ai()