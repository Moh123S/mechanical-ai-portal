import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from src.tools.vector_tool import mechanical_pdf_search

load_dotenv()

# 1. Tools & LLM
tools = [mechanical_pdf_search]
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

# 2. Tera preferred Prompt (Wahi jo pehle best chal raha tha)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior Mechanical Systems Expert. 
    - Use the 'mechanical_pdf_search' tool ONLY for technical queries.
    - If the user uses filler words or slang, ignore them and focus on the technical question.
    - For greetings or general talk, respond politely WITHOUT using tools."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 3. Agent Execution (Yahan niche wala logic use kiya hai)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True  # Ye crash hone se bachayega!
)

# 4. Memory & Loop
chat_history = []
print("🚀 Agentic RAG Final Mode Online! (Type 'exit' to stop)")

while True:
    user_input = input("\n👤 User: ")
    if user_input.lower() in ['exit', 'quit', 'bye']: break
    
    try:
        response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response["output"]))
        
        if len(chat_history) > 10: chat_history = chat_history[-10:]
        print(f"\n🤖 Expert: {response['output']}")
        
    except Exception as e:
        print(f"\n🤖 Expert: Bhai, thoda technical glitch hai, par main yahi hoon. Kya pucha tha tune?")