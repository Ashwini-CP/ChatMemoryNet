from flask import Flask, request, jsonify
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

import os

app = Flask(__name__)

# -----------------------------
# Load local model
# -----------------------------
print("⏳ Loading local model (distilbart-cnn-12-6)...")

local_pipeline = pipeline(
    "text2text-generation",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1   # -1 = CPU, 0 = GPU if available
)

llm = HuggingFacePipeline(pipeline=local_pipeline)

# -----------------------------
# Embeddings + FAISS Knowledge Base
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# A small health knowledge base
health_knowledge = [
    "Fever: Take rest, drink warm fluids, and monitor your temperature.",
    "Headache: Drink water, rest in a quiet room, and avoid stress.",
    "Cold: Drink warm fluids, inhale steam, and take rest.",
    "Cough: Drink warm water, avoid cold drinks, and consider honey with warm water.",
    "Stomach pain: Take rest, drink warm water, and avoid spicy food."
]

if os.path.exists("faiss_index"):
    print("📂 Loading FAISS index from disk...")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    print("⚠️ No FAISS index found, creating with health knowledge base...")
    vectorstore = FAISS.from_texts(health_knowledge, embeddings)
    vectorstore.save_local("faiss_index")

# -----------------------------
# Memory
# -----------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------------
# Retrieval Chain
# -----------------------------
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
)

# -----------------------------
# API Routes
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").lower().strip()
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # --- Handle greetings manually ---
    greetings = {
        "hi": "Hello! How can I help you today?",
        "hello": "Hi there! How are you feeling?",
        "bye": "Goodbye! Take care of your health.",
        "thank you": "You're welcome! Stay healthy."
    }
    for g in greetings:
        if g in user_message:
            memory.save_context({"input": user_message}, {"output": greetings[g]})
            return jsonify({"reply": greetings[g]})

    # --- Health Q&A with retrieval ---
    try:
        result = qa.invoke({
            "input": user_message,
            "chat_history": memory.load_memory_variables({}).get("chat_history", [])
        })

        reply = result["answer"].strip()

        # Fallback if no useful answer
        if not reply or "don’t know" in reply.lower():
            reply = "I don’t know. Please consult a doctor."

        memory.save_context({"input": user_message}, {"output": reply})

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
