import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# -------------------------------
# Flask app setup
# -------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------
# Load Lightweight Local Model
# -------------------------------
print("⏳ Loading local model (distilbart-cnn-12-6)...")
local_pipeline = pipeline(
    "text2text-generation",
    model="sshleifer/distilbart-cnn-12-6",
    tokenizer="sshleifer/distilbart-cnn-12-6",
    max_length=256,       # smaller output
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=local_pipeline)

# -------------------------------
# Embeddings + VectorStore
# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("faiss_index"):
    print("📂 Loading FAISS index from disk...")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    print("⚠️ No FAISS index found, creating empty one...")
    db = FAISS.from_texts(["Hello! I am your health assistant."], embeddings)
    db.save_local("faiss_index")

retriever = db.as_retriever()

# -------------------------------
# Memory + QA Chain
# -------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

# -------------------------------
# Routes
# -------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        query = data.get("message", "")
        result = qa.invoke({"question": query})
        return jsonify({"response": result["answer"], "memory": str(memory.chat_memory.messages)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
