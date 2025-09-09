from flask import Flask, request, jsonify
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
import pandas as pd
import os, json
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
    else:
        raise ValueError("❌ CSV must contain 'symptoms', 'symptom_text', and 'solution' columns")
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
# Per-user memory & graph storage
# -----------------------------
user_memories = {}
user_graphs = {}

# -----------------------------
# Greeting handler
# -----------------------------
def handle_greetings(user_message: str, user_name: str = None):
    greetings = {
        "hi": f"Hello! May I know your name?" if not user_name else f"Hello {user_name}, how are you feeling today?",
        "hello": f"Hi there! May I know your name?" if not user_name else f"Hi {user_name}, how can I help?",
        "bye": f"Goodbye {user_name if user_name else ''}! Take care of your health.",
        "thank you": "You're welcome! Stay healthy."
    }
    return greetings.get(user_message.lower().strip(), None)

# -----------------------------
# Main chat orchestrator
# -----------------------------
def orchestrator_chat(user_message, memory, user_id):
    # Check if we already know the username
    graph = user_graphs[user_id]
    user_name = graph.get("user_name")

    # Detect if user provides their name
    if user_message.lower().startswith("my name is"):
        user_name = user_message[10:].strip().capitalize()
        graph["user_name"] = user_name
        return f"Nice to meet you, {user_name}! How are you feeling today?"

    # Greetings
    greeting_reply = handle_greetings(user_message, user_name)
    if greeting_reply:
        return greeting_reply

    # Symptom direct match
    for key, solution in symptom_solution_map.items():
        if key in user_message.lower():
            if user_name:
                return f"{solution} Take care, {user_name}."
            return solution

    # FAISS search
    docs = vectorstore.similarity_search(user_message, k=1)
    if docs:
        retrieved_key = docs[0].page_content.lower().strip()
        solution = symptom_solution_map.get(retrieved_key, None)
        if solution:
            if user_name:
                return f"{solution} Take care, {user_name}."
            return solution

    # Fallback
    if user_name:
        return f"I’m not sure, {user_name}. Please consult a doctor."
    return "I don’t know. Please consult a doctor."

# -----------------------------
# Chat endpoint
# -----------------------------
@app.post("/chat")
async def chat_endpoint(item: ChatItem):
    user_id = item.user_id
    message = item.message.strip()

    # Ensure user state
    if user_id not in user_states:
        user_states[user_id] = {"name": None}

    state = user_states[user_id]
    reply = ""

    # --- Check if name already stored ---
    if state["name"] is None:
        # Try to extract a name from user input
        lowered = message.lower()
        if any(kw in lowered for kw in ["my name is", "i am", "this is"]):
            # Extract name (take last word as name)
            name = message.split()[-1].capitalize()
            state["name"] = name
            reply = f"Nice to meet you, {name}! How can I help you today?"
        elif len(message.split()) == 1 and message.isalpha():
            # If single word (like 'Ashwini'), assume it's a name
            state["name"] = message.capitalize()
            reply = f"Hello {state['name']}! Tell me how you're feeling."
        else:
            reply = "Hello! May I know your name?"
    else:
        # --- Handle symptoms normally ---
        # (Your symptom detection logic here)
        if "back pain" in message.lower():
            reply = f"Back pain irundhaal, light food sapdunga heavy food avoid pannunga. Take care, {state['name']}."
        elif "fever" in message.lower():
            reply = f"Fever irundhaal, water adhigama kudichunga rest eduthunga. Take care, {state['name']}."
        else:
            reply = f"I’m not sure, {state['name']}. Please consult a doctor."

    # Update graph (store conversation)
    G = user_graphs[user_id]
    G.add_node(message, type="user_message")
    G.add_node(reply, type="bot_reply")
    G.add_edge(message, reply)

    return {"reply": reply}

# -----------------------------
# Graph visualization
# -----------------------------
@app.route("/graphviz/<user_id>", methods=["GET"])
def get_graph_viz(user_id):
    graph_path = f"storage/graph_{user_id}.json"
    if not os.path.exists(graph_path):
        return "No graph available for this user."

    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    G = nx.DiGraph()
    for i, node in enumerate(graph_data["nodes"]):
        node_id = f"{node['role']}_{i}"
        G.add_node(node_id, label=f"{node['role']}: {node['content'][:40]}")
        if i > 0:
            prev_id = f"{graph_data['nodes'][i-1]['role']}_{i-1}"
            G.add_edge(prev_id, node_id)

    net = Network(height="500px", width="100%", directed=True)
    net.from_nx(G)
    net.save_graph(f"storage/chat_graph_{user_id}.html")

    with open(f"storage/chat_graph_{user_id}.html", "r", encoding="utf-8") as f:
        html = f.read()
    return html

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
