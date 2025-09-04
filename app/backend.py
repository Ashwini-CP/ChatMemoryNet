from flask import Flask, request, jsonify
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Use new recommended imports
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
    device=-1   # CPU (-1), use 0 for GPU
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
# Custom QA Prompt
# -----------------------------
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful Health Assistant bot.
Use the provided context to answer the user's question in simple, clear language.
If the answer is not in the context, reply: "I don’t know. Please consult a doctor."

Context:
{context}

Question:
{question}

Answer:"""
)

# -----------------------------
# Retrieval Chain with custom prompt
# -----------------------------
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
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
