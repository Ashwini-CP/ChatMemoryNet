from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

app = Flask(__name__)

# -----------------------------
# Embeddings + FAISS
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Predefined symptom-advice list
texts = [
    # English
    "fever: Rest, drink fluids, monitor temperature. Take paracetamol if needed. See a doctor if fever persists >3 days.",
    "headache: Rest, stay hydrated, avoid bright lights. Take OTC painkillers if needed.",
    "cold and cough: Drink warm water, rest, use a humidifier. See a doctor if symptoms worsen.",
    "stomach pain: Eat light meals, drink warm water, consult a doctor if severe.",
    # Tanglish
    "ennaikku fever irukku: Rest pannunga, thanni kudunga, paracetamol kudikalam. Fever 3 naal mela irundha doctor kitta poonga.",
    "ennaikku headache irukku: Rest pannunga, thanni kudunga, bright light avoid pannunga. OTC painkiller edunga.",
    "kodu cold um cough um irukku: Warm water kudunga, rest pannunga, humidifier use pannunga. Symptoms adhigam irundha doctor kitta poonga.",
    "stomach pain: Heavy meals avoid pannunga, warm water kudunga, severe aana doctor kitta poonga."
]

# Load or create FAISS index
if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local("faiss_index")

# -----------------------------
# API Route
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Search FAISS for closest match
        docs = vectorstore.similarity_search(user_message, k=1)
        if docs:
            # Return advice from closest match
            answer = docs[0].page_content.split(":", 1)[1].strip() if ":" in docs[0].page_content else docs[0].page_content
        else:
            # Fallback general advice
            answer = "Rest well, stay hydrated, and consult a doctor if symptoms persist."

        return jsonify({"reply": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
