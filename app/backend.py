from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
import pandas as pd
import os

app = Flask(__name__)

# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "data/healthcare_tanglish_dataset (2).csv"
df = pd.read_csv(DATA_PATH)

# Ensure lowercase for consistency
df["symptom"] = df["symptom"].astype(str).str.lower().str.strip()
df["symptom_text"] = df["symptom_text"].astype(str).str.lower().str.strip()
df["solution"] = df["solution"].astype(str).str.strip()

# Build mapping: symptom_text → {symptom, solution}
symptom_solution_map = {
    row["symptom_text"]: {
        "symptom": row["symptom"],
        "solution": row["solution"]
    }
    for _, row in df.iterrows()
}

# -----------------------------
# Load / Build FAISS Index
# -----------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
symptom_texts = df["symptom_text"].tolist()

if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_texts(symptom_texts, embeddings)
    vectorstore.save_local("faiss_index")

# -----------------------------
# Chat Memory & Profiles
# -----------------------------
memory = ConversationBufferMemory(return_messages=True)
user_profiles = {}

def get_user_profile(user_id):
    if user_id not in user_profiles:
        user_profiles[user_id] = {"name": None, "symptoms": []}
    return user_profiles[user_id]

# -----------------------------
# Name Extraction
# -----------------------------
def extract_name(text):
    text = text.lower()
    if text.startswith("i am "):
        return text.replace("i am ", "").strip().title()
    if text.startswith("im "):
        return text.replace("im ", "").strip().title()
    if text.startswith("my name is "):
        return text.replace("my name is ", "").strip().title()
    if len(text.split()) == 1 and text.isalpha():
        return text.title()
    return None

# -----------------------------
# Symptom Preprocessing
# -----------------------------
def preprocess_text(text):
    return text.lower().strip()

# -----------------------------
# Chat Endpoint
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_id = request.json.get("user_id", "default")
    user_message = request.json.get("message", "").strip()

    profile = get_user_profile(user_id)

    # Step 1: Collect name
    if profile["name"] is None:
        possible_name = extract_name(user_message)
        if possible_name:
            profile["name"] = possible_name
            reply = f"Nice to meet you, {profile['name']}! What symptoms are you facing today?"
        else:
            reply = "I didn’t catch your name. Could you tell me your name?"
        memory.chat_memory.add_ai_message(reply)
        return jsonify({"reply": reply})

    # Step 2: Collect symptoms and retrieve solution
    cleaned_message = preprocess_text(user_message)
    docs = vectorstore.similarity_search(cleaned_message, k=1)

    reply_data = {
        "symptom": None,
        "symptom_text": None,
        "solution": "I don’t know. Please consult a doctor."
    }

    if docs:
        retrieved_key = docs[0].page_content.lower().strip()
        if retrieved_key in symptom_solution_map:
            mapped = symptom_solution_map[retrieved_key]
            reply_data = {
                "symptom": mapped["symptom"],
                "symptom_text": retrieved_key,
                "solution": mapped["solution"]
            }
            profile["symptoms"].append(mapped["symptom"])

    memory.chat_memory.add_user_message(user_message)
    memory.chat_memory.add_ai_message(reply_data["solution"])

    return jsonify(reply_data)

# -----------------------------
# Graph Endpoint
# -----------------------------
@app.route("/graph", methods=["GET"])
def get_graph_json():
    history = []
    for msg in memory.chat_memory.messages:
        history.append({"role": msg.type, "content": msg.content})

    profiles = {uid: profile for uid, profile in user_profiles.items()}
    return jsonify({"chat_history": history, "profiles": profiles})


if __name__ == "__main__":
    app.run(debug=True)
