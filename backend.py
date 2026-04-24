import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# 1. Setup Models & Retriever
def get_rag_chain():
    llm = ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_name = "mechanical-ai-portal"
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # 2. History Aware Retriever Logic
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "formulate a standalone question. Do NOT answer it, just reformulate it."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # 3. QA Chain Logic
    system_prompt = (
        "You are a Senior Mechanical Engineer Mentor. Use context to answer in Hinglish. "
        "Keep technical terms in English. Context:\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Final RAG Chain
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Exporting the chain
rag_chain = get_rag_chain()