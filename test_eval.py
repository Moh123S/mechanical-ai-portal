import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

load_dotenv()

# 1. Setup Models
llm = ChatGroq(model_name="llama-3.1-8b-instant")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name="mechanical-ai-portal", embedding=embeddings)

# 2. Basic QA Chain for Testing
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 3. Test Questions (Apne PDF ke hisab se badal sakte ho)
questions = [
    "What is the main function of the centrifugal pump?",
    "What are the maintenance steps for the impeller?",
    "What causes cavitation in the system?"
]

# 4. Collecting Results
data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

print("🧪 Testing shuru ho rahi hai...")

for query in questions:
    result = qa_chain.invoke(query)
    data["question"].append(query)
    data["answer"].append(result["result"])
    # Context nikalne ke liye
    docs = vectorstore.similarity_search(query)
    data["contexts"].append([doc.page_content for doc in docs])
    # Ground Truth (Jo asali answer hona chahiye - manual se dekh kar likho)
    data["ground_truth"].append("Centrifugal pump moves fluid by rotational energy.") 

# 5. Dataset Convert & Evaluate
dataset = Dataset.from_dict(data)
# Note: Ragas ko OpenAI key chahiye hoti hai scoring ke liye, 
# par hum manually bhi dekh sakte hain output.
print("\n--- TEST RESULTS ---")
for i in range(len(questions)):
    print(f"\nQ: {data['question'][i]}")
    print(f"AI Answer: {data['answer'][i]}")