import streamlit as st
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

# Page Config
st.set_page_config(page_title="Mechanical AI Portal", page_icon="⚙️")
st.title("⚙️ Mechanical AI Portal")
st.markdown("---")

# 1. Setup Models (Caching for speed)
@st.cache_resource
def load_models():
    llm = ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_name = "mechanical-ai-portal"
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return llm, vectorstore.as_retriever()

llm, retriever = load_models()

# 2. Logic: History Aware Retriever & QA Chain
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
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 3. Streamlit Session State (Chat History)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat History
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat Input
if user_input := st.chat_input("Pucho Mohit bhai..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("AI dimaag laga raha hai..."):
            response = rag_chain.invoke({
                "input": user_input, 
                "chat_history": st.session_state.chat_history
            })
            st.markdown(response["answer"])
    
    # Update History
    st.session_state.chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=response["answer"]),
    ])