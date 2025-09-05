from flask import Flask, request, jsonify
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
import os

app = Flask(__name__)

# -----------------------------
# Load Local Model
# -----------------------------
print("⏳ Loading local model (distilbart-cnn-12-6)...")
local_pipeline = pipeline(
    "text2text-generation",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1  # CPU; use 0 for GPU
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
    print("⚠️ No FAISS index found, creating empty one...")
    vectorstore = FAISS.from_texts(["Welcome to Health Memory Bot!"], embeddings)
    vectorstore.save_local("faiss_index")

# -----------------------------
# Memory
# -----------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------------
# Custom Prompt (new format)
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are a helpful Health Assistant chatbot.

Rules:
- If the user greets (hi, hello, bye, thank you), reply politely in simple English.
- Use the provided context to answer health-related questions clearly.
- If the answer is NOT in the context, reply exactly: "I don’t know. Please consult a doctor."
- If the user says something unrelated to health and no context applies, reply: "I am your health assistant."

Chat History:
{chat_history}

Context:
{context}

Question:
{input}

Answer:
""")

# -----------------------------
# Build New QA Chain
# -----------------------------
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa = create_retrieval_chain(retriever, document_chain)

# -----------------------------
# API Route
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        result = qa.invoke({"input": user_message, "chat_history": memory.load_memory_variables({}).get("chat_history", [])})
        
        # Save interaction in memory
        memory.save_context({"input": user_message}, {"output": result["answer"]})
        
        return jsonify({"reply": result["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
