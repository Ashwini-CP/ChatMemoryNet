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
# Load HuggingFace model locally
# -------------------------------
print("⏳ Loading local model (google/flan-t5-large)...")
local_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",  # change to flan-t5-xl if you have more RAM
    tokenizer="google/flan-t5-large",
    max_length=512,
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
    from langchain_community.vectorstores import FAISS
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
