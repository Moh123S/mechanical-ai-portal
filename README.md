# ⚙️ Mechanical AI Portal: Agentic RAG System

An advanced AI-powered assistant designed for Mechanical Engineers to interact with technical manuals, maintenance logs, and engineering procedures using **Agentic Reasoning**.

## 🚀 Core Features
- **Agentic RAG:** Not just a chatbot; it's an AI Agent that decides when to search the manual and when to use its own knowledge.
- **Hybrid Search:** Integrated with **Pinecone Vector DB** for high-precision technical retrieval.
- **Memory Management:** Remembers previous conversation context for seamless troubleshooting.
- **Error Handling:** Built-in parsing safety to handle complex engineering queries without crashing.

## 🛠️ Tech Stack
- **LLM:** Groq (Llama-3.1-8B-Instant)
- **Framework:** LangChain (Agentic Workflow)
- **Vector DB:** Pinecone
- **Embeddings:** HuggingFace (Sentence Transformers)
- **UI:** Streamlit (For Web Access)

## 📂 Project Structure
- `src/tools/`: Custom tools for the Agent (Vector Search, etc.)
- `agent_test.py`: Terminal-based testing module for Agentic Reasoning.
- `app.py`: Main Streamlit application.

## 🔧 Installation
1. Clone the repo: `git clone https://github.com/Moh123S/mechanical-ai-portal.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Setup `.env` with GROQ_API_KEY and PINECONE_API_KEY.
4. Run: `python agent_test.py`