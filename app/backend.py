from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
import pandas as pd
import os, re, uuid
import networkx as nx
from pyvis.network import Network

# ===============================
# 🔧 Flask Setup
# ===============================
app = Flask(__name__)
CORS(app)

# ===============================
# 🧠 Load Local Model (Summarizer / Generator)
# ===============================
print("⏳ Loading local model (distilbart-cnn-12-6)...")
local_pipeline = pipeline(
    "text2text-generation",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1
)
llm = HuggingFacePipeline(pipeline=local_pipeline)

# ===============================
# 📂 Load Healthcare Dataset
# ===============================
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
    else:
        raise ValueError("❌ CSV must contain 'symptoms', 'symptom_text', and 'solution'")
else:
    print("⚠️ Dataset not found, using fallback dictionary...")
    symptom_solution_map = {
        "fever": "Take rest, drink warm fluids, and monitor your temperature.",
        "headache": "Drink water, rest in a quiet room, and avoid stress.",
        "cold": "Drink warm fluids, inhale steam, and take rest.",
        "cough": "Drink warm water, avoid cold drinks, and consider honey with warm water.",
        "stomach pain": "Take rest, drink warm water, and avoid spicy food."
    }
    health_knowledge = list(symptom_solution_map.keys())

# ===============================
# 🔍 FAISS Vector Store
# ===============================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("faiss_index"):
    print("📂 Loading FAISS index from disk...")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    print("⚠️ No FAISS index found, creating new one...")
    vectorstore = FAISS.from_texts(health_knowledge, embeddings)
    vectorstore.save_local("faiss_index")

# ===============================
# 🧾 Global Graph & User States
# ===============================
global_graph = nx.DiGraph()
user_states = {}   # { user_name: {"history": [], "name": str} }

# ===============================
# 🧾 Utility: Extract Name
# ===============================
def extract_name(message: str) -> str:
    lowered = message.lower()
    match = re.search(r"(?:my name is|i am|this is)\s+([a-zA-Z]+)", lowered)
    if match:
        return match.group(1).capitalize()
    words = message.strip().split()
    if len(words) == 1 and words[0].isalpha():
        return words[0].capitalize()
    return None

# ===============================
# 💬 Chat Orchestrator
# ===============================
def orchestrator_chat(user_id, message):
    state = user_states[user_id]

    # Handle name if not set
    if state["name"] is None:
        name = extract_name(message)
        if name:
            state["name"] = name
            return f"Nice to meet you, {name}! How can I help you today?"
        else:
            return "Hello! May I know your name?"

    user_name = state["name"]  # <-- use extracted name, not user_id

    # Symptom direct match
    for key, solution in symptom_solution_map.items():
        if key in message.lower():
            return f"{solution} Take care, {user_name}."

    # FAISS semantic search
    docs = vectorstore.similarity_search(message, k=1)
    if docs:
        retrieved = docs[0].page_content.lower().strip()
        solution = symptom_solution_map.get(retrieved, None)
        if solution:
            return f"{solution} Take care, {user_name}."

    return f"I’m not sure, {user_name}. Please consult a doctor."


# ===============================
# 📌 Chat Endpoint
# ===============================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "").strip()

    # Extract user name
    user_name = data.get("user_id") or extract_name(message) or f"Anonymous-{str(uuid.uuid4())[:6]}"

    # Create root node if new user
    if not global_graph.has_node(user_name):
        global_graph.add_node(user_name, type="user_root")

    # Add conversation nodes
    user_node = f"user: {message}"
    bot_reply = orchestrator_chat(user_name, message)
    bot_node = f"bot: {bot_reply}"

    global_graph.add_node(user_node, type="user_message")
    global_graph.add_node(bot_node, type="bot_reply")
    global_graph.add_edge(user_name, user_node)
    global_graph.add_edge(user_node, bot_node)

    # Save user history
    user_states[user_name]["history"].append({"user": message, "bot": bot_reply})

    return jsonify({"user_id": user_name, "reply": bot_reply})

# ===============================
# 📊 Graphviz Endpoint
# ===============================
@app.route("/graphviz/<user_id>", methods=["GET"])
def get_graph_viz(user_id):
    if not global_graph.has_node(user_id):
        return "No graph available for this user."

    net = Network(height="600px", width="100%", directed=True)
    # Show only user's subtree
    user_nodes = list(nx.descendants(global_graph, user_id)) + [user_id]
    subgraph = global_graph.subgraph(user_nodes)
    net.from_nx(subgraph)
    return net.generate_html()

# ===============================
# 🚀 Run
# ===============================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
