import streamlit as st
from backend import rag_chain
from langchain_core.messages import HumanMessage, AIMessage

# Page Config
st.set_page_config(page_title="Mechanical AI Portal", page_icon="⚙️", layout="wide")

st.title("⚙️ Mechanical AI Portal")
st.markdown("---")

# Session State for History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Info
with st.sidebar:
    st.header("Project Info")
    st.write("Sector: Mechanical Engineering")
    st.write("System: Advanced RAG")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Display Chat History
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat Input
if user_input := st.chat_input("Pucho Mohit bhai (e.g., Pump maintenance steps?)..."):
    # User message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Assistant message
    with st.chat_message("assistant"):
        with st.spinner("AI Engineer is thinking..."):
            response = rag_chain.invoke({
                "input": user_input, 
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]
            st.markdown(answer)
    
    # Update History
    st.session_state.chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=answer),
    ])