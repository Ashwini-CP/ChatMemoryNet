from flask import Flask, request, jsonify
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
import pandas as pd
import os
import networkx as nx
from pyvis.network import Network

app = Flask(__name__)

# -----------------------------
# Load local model
# -----------------------------
print("⏳ Loading local model (distilbart-cnn-12-6)...")
local_pipeline = pipeline(
    "text2text-generation",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1
)
llm = HuggingFacePipeline(pipeline=local_pipeline)

# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "data/healthcare_tanglish_dataset (2).csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    if "symptoms" in df.columns and "symptom_text" in df.columns and "solution" in df.columns:
        symptom_solution_map = {}
        knowledge_entries = []

        for _, row in df.iterrows():
            symptom = str(row["symptoms"]).lower().strip()
            symptom_text = str(row["symptom_text"]).lower().strip()
            solution = str(row["solution"]).strip()

            symptom_solution_map[symptom] = solution
            symptom_solution_map[symptom_text] = solution
            knowledge_entries.extend([symptom, symptom_text])

        health_knowledge = list(set(knowledge_entries))
        print(f"✅ Loaded dataset with {len(df)} records")
    else:
        raise ValueError("❌ CSV must contain 'symptoms', 'symptom_text', and 'solution' columns")
else:
    symptom_solution_map = {
        "fever": "Take rest, drink warm fluids, and monitor your temperature.",
        "headache": "Drink water, rest in a quiet room, and avoid stress.",
        "cold": "Drink warm fluids, inhale steam, and take rest.",
        "cough": "Drink warm water, avoid cold drinks, and consider honey with warm water.",
        "stomach pain": "Take rest, drink warm water, and avoid spicy food."
    }
    health_knowledge = list(symptom_solution_map.keys())
    print("⚠️ Dataset not found, using fallback dictionary...")

# -----------------------------
# Build FAISS
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
user_name = None   # store the user’s name globally

# -----------------------------
# Greeting handler
# -----------------------------
def handle_greetings(user_message: str):
    greetings = {
        "hi": "Hello! May I know your name?",
        "hello": "Hi there! What’s your name?",
        "bye": "Goodbye! Take care of your health.",
        "thank you": "You're welcome! Stay healthy."
    }
    return greetings.get(user_message.lower().strip(), None)

# -----------------------------
# Chat endpoint
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    global user_name

    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Save user message to memory
    memory.chat_memory.add_user_message(user_message)

    # ----------------------
    # If user_name not set → ask for name
    # ----------------------
    if not user_name:
        if user_message.lower().startswith(("hi", "hello")):
            reply = "Hello! May I know your name?"
        elif user_message.lower().startswith("my name is"):
            user_name = user_message.replace("my name is", "").strip().title()
            reply = f"Nice to meet you, {user_name}! Tell me what’s troubling you today?"
        else:
            reply = "I don’t know your name yet. Could you please tell me your name? (e.g., My name is Arun)"
        
        memory.chat_memory.add_ai_message(reply)
        return jsonify({"reply": reply})

    # ----------------------
    # Greetings
    # ----------------------
    greeting_reply = handle_greetings(user_message)
    if greeting_reply:
        memory.chat_memory.add_ai_message(greeting_reply)
        return jsonify({"reply": greeting_reply})

    try:
        # 1️⃣ Direct match
        for key, solution in symptom_solution_map.items():
            if key in user_message.lower():
                personalized = f"{solution} Take care, {user_name}."
                memory.chat_memory.add_ai_message(personalized)
                return jsonify({"reply": personalized})

        # 2️⃣ FAISS similarity search
        docs = vectorstore.similarity_search(user_message, k=1)
        if docs:
            retrieved_key = docs[0].page_content.lower().strip()
            solution = symptom_solution_map.get(retrieved_key, None)
            if solution:
                personalized = f"{solution} Take care, {user_name}."
                memory.chat_memory.add_ai_message(personalized)
                return jsonify({"reply": personalized})

        # 3️⃣ Fallback
        reply = f"I’m not sure, {user_name}. Please consult a doctor."
        memory.chat_memory.add_ai_message(reply)
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Graph JSON
# -----------------------------
@app.route("/graph", methods=["GET"])
def get_graph_json():
    history = []
    for msg in memory.chat_memory.messages:
        history.append({"role": msg.type, "content": msg.content})
    return jsonify({"chat_history": history})

# -----------------------------
# Graph Visualization
# -----------------------------
@app.route("/graphviz", methods=["GET"])
def get_graph_viz():
    G = nx.DiGraph()

    # Create nodes for each message
    for i, msg in enumerate(memory.chat_memory.messages):
        node_id = f"{msg.type}_{i}"
        G.add_node(node_id, label=f"{msg.type}: {msg.content[:40]}")

        if i > 0:
            prev_id = f"{memory.chat_memory.messages[i-1].type}_{i-1}"
            G.add_edge(prev_id, node_id)

    # Build interactive graph
    net = Network(height="500px", width="100%", directed=True)
    net.from_nx(G)
    net.save_graph("chat_graph.html")

    with open("chat_graph.html", "r", encoding="utf-8") as f:
        html = f.read()

    return html

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
