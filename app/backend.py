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
print("⏳ Loading FLAN-T5 conversational model...")
local_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=128,
    truncation=True,
    device=-1  # CPU, change to 0 for GPU
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
    print("⚠️ No FAISS index found, creating one with basic health knowledge...")
    texts = [
        "fever and headache: Take rest, drink fluids, and monitor your temperature. Consult a doctor if it persists.",
        "cold and cough: Drink warm water, rest, and take over-the-counter medicine if needed.",
        "stomach pain: Avoid heavy meals, drink warm water, and consult a doctor if severe.",
        "general: I am a health assistant. I can give basic advice on common symptoms and wellness tips."
    ]
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local("faiss_index")

# -----------------------------
# Memory
# -----------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------------
# Prompt Template
# -----------------------------
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful health assistant.
Answer the user’s question in simple, clear language. Use Tanglish or English based on the user's message.

Context:
{context}

User: {question}
Bot:"""
)

# -----------------------------
# Conversational Retrieval Chain
# -----------------------------
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

# -----------------------------
# API Route
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Use invoke() instead of __call__ to avoid deprecated warnings
        result = qa.invoke({"question": user_message})
        response_text = result.get("answer", "").strip()

        # Fallback if empty
        if not response_text:
            llm_result = llm.invoke({"text": user_message})
            response_text = llm_result.get("text", "").strip()

        return jsonify({"reply": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
