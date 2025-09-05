from flask import Flask, request, jsonify
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

import os

app = Flask(__name__)

# -----------------------------
# Load conversational model
# -----------------------------
print("⏳ Loading local conversational model (flan-t5-base)...")

local_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256,
    device=-1   # CPU (set to 0 if you have GPU)
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
    print("⚠️ No FAISS index found. Creating with sample health knowledge...")
    texts = [
        "Symptom: fever, headache. Solution: Take rest, drink warm fluids, and use paracetamol if needed.",
        "Symptom: cold and cough. Solution: Drink warm water, take steam inhalation, and rest well.",
        "Symptom: stomach pain. Solution: Eat light food, drink water, and avoid spicy food.",
        "Symptom: stress or anxiety. Solution: Try meditation, deep breathing, and adequate sleep.",
        "Symptom: body pain. Solution: Take rest, apply warm compress, and stay hydrated."
    ]
    vectorstore = FAISS.from_texts(texts, embeddings)
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
    template=(
        "You are a helpful health assistant.\n\n"
        "Use the following context to answer the user in simple, clear language.\n"
        "If the answer is not in the context, reply exactly: I don’t know. Please consult a doctor.\n\n"
        "Context:\n{context}\n\n"
        "User: {question}\n\n"
        "Chat History: You should remember previous chats to make replies consistent.\n\n"
        "Assistant:"
    )
)

# -----------------------------
# Retrieval Chain
# -----------------------------
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    return_source_documents=False
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
        result = qa.invoke({"question": user_message})
        return jsonify({"reply": result["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
