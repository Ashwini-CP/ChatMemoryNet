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
user_states = {}   # stores {user_id: {"name": str}}
user_graphs = {}   # stores conversation graphs


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
# Chat endpoint
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    user_id = data.get("user_id", "default_user")
    message = data.get("message", "").strip()

    # Ensure user state
    if user_id not in user_states:
        user_states[user_id] = {"name": None}
        user_graphs[user_id] = nx.DiGraph()

    state = user_states[user_id]
    reply = ""

    # --- Check if name already stored ---
    if state["name"] is None:
        lowered = message.lower()
        if any(kw in lowered for kw in ["my name is", "i am", "this is"]):
            # Extract last word as name
            name = message.split()[-1].capitalize()
            state["name"] = name
            reply = f"Nice to meet you, {name}! How can I help you today?"
        elif len(message.split()) == 1 and message.isalpha():
            # If single word, assume it's a name
            state["name"] = message.capitalize()
            reply = f"Hello {state['name']}! Tell me how you're feeling."
        else:
            reply = "Hello! May I know your name?"
    else:
        # --- Handle greetings ---
        greeting_reply = handle_greetings(message, state["name"])
        if greeting_reply:
            reply = greeting_reply
        else:
            # --- Handle symptoms ---
            found = False
            for key, solution in symptom_solution_map.items():
                if key in message.lower():
                    reply = f"{solution} Take care, {state['name']}."
                    found = True
                    break

            if not found:
                # --- FAISS search ---
                docs = vectorstore.similarity_search(message, k=1)
                if docs:
                    retrieved_key = docs[0].page_content.lower().strip()
                    solution = symptom_solution_map.get(retrieved_key, None)
                    if solution:
                        reply = f"{solution} Take care, {state['name']}."
                    else:
                        reply = f"I’m not sure, {state['name']}. Please consult a doctor."
                else:
                    reply = f"I’m not sure, {state['name']}. Please consult a doctor."

    # Update graph (store conversation)
    G = user_graphs[user_id]
    G.add_node(message, type="user_message")
    G.add_node(reply, type="bot_reply")
    G.add_edge(message, reply)

    # Save graph as JSON
    os.makedirs("storage", exist_ok=True)
    graph_path = f"storage/graph_{user_id}.json"
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": [{"role": "user", "content": message},
                             {"role": "bot", "content": reply}]}, f, indent=2)

    return jsonify({"reply": reply})


# -----------------------------
# Graph visualization
# -----------------------------
@app.route("/graphviz/<user_id>", methods=["GET"])
def get_graph_viz(user_id):
    if user_id not in user_graphs:
        return "No graph available for this user."

    G = user_graphs[user_id]

    net = Network(height="500px", width="100%", directed=True)
    net.from_nx(G)

    os.makedirs("storage", exist_ok=True)
    graph_file = f"storage/chat_graph_{user_id}.html"
    net.save_graph(graph_file)

    with open(graph_file, "r", encoding="utf-8") as f:
        html = f.read()
    return html


# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
