from flask import Flask, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub  # or OpenAI
from langchain.memory import ConversationBufferMemory
import pandas as pd

app = Flask(__name__)

# ===== 1. Load dataset and create embeddings =====
df = pd.read_csv("data/health_tanglish_elaborated.csv")
symptoms = df["symptoms"].astype(str).tolist()
solutions = df["solution"].astype(str).tolist()

# Create text pairs (symptom + solution)
docs = [f"Symptom: {s}\nSolution: {sol}" for s, sol in zip(symptoms, solutions)]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(docs, embeddings)

# ===== 2. Setup Memory =====
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ===== 3. Define LLM (you can also use OpenAI API here) =====
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.2, "max_length":256})

# ===== 4. Conversational Retrieval Chain =====
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("message", "")
    
    if not query:
        return jsonify({"reply": "Please enter a symptom or question."})
    
    result = qa.run(query)
    return jsonify({"reply": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
