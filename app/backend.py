from flask import Flask, request, jsonify
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# New recommended imports
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

import os

app = Flask(__name__)

# -----------------------------
# Load Local LLM
# -----------------------------
print("⏳ Loading local model (flan-t5-large)...")

local_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=-1   # CPU, set to 0 for GPU
)

llm = HuggingFacePipeline(pipeline=local_pipeline)

# -----------------------------
# Embeddings + FAISS
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("faiss_index"):
    print("📂 Loading FAISS index from disk...")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    print("⚠️ No FAISS index found, creating a new one...")

    health_texts = [
        "Symptom: fever, headache. Solution: Take rest, drink warm fluids, and use paracetamol.",
        "Tanglish: enakku fever irukku. Solution: Take rest, drink warm fluids, and use paracetamol.",
        "Symptom: cough and cold. Solution: Steam inhalation, drink warm water, and rest well.",
        "Tanglish: enakku thalai vali irukku. Solution: Take rest, drink water, and avoid stress.",
        "Symptom: stomach pain. Solution: Take light food, drink warm water, and consult a doctor if severe.",
    ]

    vectorstore = FAISS.from_texts(health_texts, embeddings)
    vectorstore.save_local("faiss_index")

# -----------------------------
# Memory
# -----------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------------
# Custom Prompt
# -----------------------------
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful Health Assistant.

Chat history and context are below:
{context}

User question: {question}

Rules:
- If the user greets (hi, hello, bye, thank you), respond politely in simple English.
- If the answer is in the health context, reply clearly with the solution.
- If the answer is NOT in context, reply exactly: "I don’t know. Please consult a doctor."
- If no health question is asked, reply: "I am your health assistant."
""",
)

# -----------------------------
# Retrieval Chain
# -----------------------------
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    return_source_documents=False,
)

# -----------------------------
# API Routes
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = qa.run(user_message)
        return jsonify({"reply": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
