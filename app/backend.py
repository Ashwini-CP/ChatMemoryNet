from flask import Flask, request, jsonify
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
import os

app = Flask(__name__)

# -----------------------------
# Load FLAN-T5 model
# -----------------------------
print("⏳ Loading FLAN-T5 model...")
model_name = "google/flan-t5-base"  # Use flan-t5-large if possible

local_pipeline = pipeline(
    "text2text-generation",
    model=model_name,
    max_length=128,
    truncation=True,
    device=-1  # CPU, use 0 for GPU
)
llm = HuggingFacePipeline(pipeline=local_pipeline)

# -----------------------------
# Embeddings + FAISS retriever
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    # Create FAISS index with Tanglish + English health advice
    texts = [
        "fever: Rest, drink fluids, monitor temperature. Take paracetamol if needed. See a doctor if fever persists >3 days.",
        "headache: Rest, stay hydrated, avoid bright lights. Take OTC painkillers if needed.",
        "cold and cough: Drink warm water, rest, use a humidifier. See a doctor if symptoms worsen.",
        "stomach pain: Eat light meals, drink warm water, consult a doctor if severe.",
        "ennaikku fever irukku: Rest pannunga, thanni kudunga, paracetamol kudikalam. Fever 3 naal mela irundha doctor kitta poonga.",
        "ennaikku headache irukku: Rest pannunga, thanni kudunga, bright light avoid pannunga. OTC painkiller edunga.",
        "kodu cold um cough um irukku: Warm water kudunga, rest pannunga, humidifier use pannunga. Symptoms adhigam irundha doctor kitta poonga.",
        "stomach pain: Heavy meals avoid pannunga, warm water kudunga, severe aana doctor kitta poonga."
    ]
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local("faiss_index")

# -----------------------------
# Memory: last 5 messages
# -----------------------------
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True
)

# -----------------------------
# Prompt: Always give actionable advice
# -----------------------------
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and empathetic health assistant.
- Respond in the same language as the user (Tanglish or English).
- Always give clear, concise, actionable advice about symptoms.
- Ignore unknown words, do not explain them.
- Never ask questions back.
- Use context only if relevant.
- Focus on the latest user message.
- Avoid repeating sentences.
- If unsure, give general advice for common symptoms.

Context:
{context}

User: {question}
Bot:"""
)

# -----------------------------
# Retrieval Chain
# -----------------------------
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k":2}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

# -----------------------------
# API route
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    try:
        result = qa.invoke({"question": user_message})
        response = result.get("answer") or result.get("result") or ""
        response = response.strip()
        if not response:
            # fallback LLM
            llm_result = llm.invoke({"text": user_message})
            response = llm_result.get("text", "").strip()
        return jsonify({"reply": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
