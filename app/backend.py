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
# Optional: switch to "google/flan-t5-large" for more human-like responses if you have RAM/GPU
model_name = "google/flan-t5-base"

local_pipeline = pipeline(
    "text2text-generation",
    model=model_name,
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
    print("⚠️ No FAISS index found, creating one with richer health knowledge...")
    texts = [
        # English examples
        "fever and headache: Take rest, drink fluids, and monitor your temperature. If high fever persists, consult a doctor.",
        "fever and cough: Rest well, drink plenty of fluids, use a humidifier. If fever lasts more than 3 days, see a doctor.",
        "cold: Rest, stay hydrated, and take OTC medicine if needed.",
        "stomach pain: Avoid heavy meals, drink warm water, consult a doctor if severe.",
        "general: I am a health assistant. I provide basic advice on symptoms and wellness tips.",
        # Tanglish examples
        "ennaikku fever um headache um irukku: Rest pannunga, thanni kudunga. Fever adhigam irundha doctor kitta poonga.",
        "ennaikku fever um cough um irukku: Rest pannunga, thanni kudunga, humidifier use pannunga. Fever 3 naal mela irundha doctor kitta poonga.",
        "kodu cold irukku: Rest pannunga, thanni kudunga, OTC medicine edunga if necessary.",
        "stomach pain: Heavy meals avoid pannunga, warm water kudunga, severe aana doctor kitta poonga.",
        "general: Naan unga health assistant. Ungaluku basic symptoms advice and wellness tips kuduren."
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
You are a helpful and empathetic health assistant.
- Respond in the same language and style as the user (Tanglish or English).
- Give clear, concise, actionable advice.
- Use context if relevant.
- Avoid repeating sentences; respond like a human.
- If the context has no answer, still provide helpful advice for common symptoms.

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
        # Use invoke() for LangChain v0.1+
        result = qa.invoke({"question": user_message})
        response_text = result.get("answer", "").strip()

        # Fallback to raw LLM if empty
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
