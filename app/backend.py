from flask import Flask, request, jsonify
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
import pandas as pd
import os

app = Flask(__name__)

# -----------------------------
# Load local model
# -----------------------------
print("⏳ Loading local model (distilbart-cnn-12-6)...")

local_pipeline = pipeline(
    "text2text-generation",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1  # CPU, use 0 for GPU
)

llm = HuggingFacePipeline(pipeline=local_pipeline)

# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "data/health_tanglish_elaborated.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    health_knowledge = df.apply(lambda row: f"{row['symptom_index']}: {row['solution']}", axis=1).tolist()
else:
    # fallback dataset if CSV missing
    health_knowledge = [
        "Fever: Take rest, drink warm fluids, and monitor your temperature.",
        "Headache: Drink water, rest in a quiet room, and avoid stress.",
        "Cold: Drink warm fluids, inhale steam, and take rest.",
        "Cough: Drink warm water, avoid cold drinks, and consider honey with warm water.",
        "Stomach pain: Take rest, drink warm water, and avoid spicy food."
    ]

# -----------------------------
# Embeddings + FAISS
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("faiss_index"):
    print("📂 Loading FAISS index from disk...")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    print("⚠️ No FAISS index found, creating from dataset...")
    vectorstore = FAISS.from_texts(health_knowledge, embeddings)
    vectorstore.save_local("faiss_index")

# -----------------------------
# Memory
# -----------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------------
# Conversational Retrieval Chain
# -----------------------------
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
)

# -----------------------------
# Greeting handler
# -----------------------------
def handle_greetings(user_message: str):
    greetings = {
        "hi": "Hello! How can I help you today?",
        "hello": "Hi there! How are you feeling?",
        "bye": "Goodbye! Take care of your health.",
        "thank you": "You're welcome! Stay healthy."
    }
    msg = user_message.lower().strip()
    return greetings.get(msg, None)

# -----------------------------
# API Routes
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Check greetings
    greeting_reply = handle_greetings(user_message)
    if greeting_reply:
        return jsonify({"reply": greeting_reply})

    try:
        response = qa.invoke({"question": user_message})
        answer = response.get("answer", "").strip()

        if not answer or "use the following" in answer.lower():
            answer = "I don’t know. Please consult a doctor."

        return jsonify({"reply": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
