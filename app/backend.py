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
# Load conversational model
# -----------------------------
print("⏳ Loading FLAN-T5 conversational model...")
# Optional: switch to "google/flan-t5-large" for more coherent responses
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
    print("⚠️ No FAISS index found, creating FAISS index with health knowledge...")
    texts = [
        # English symptom-advice pairs
        "fever: Rest, drink fluids, monitor temperature. Use paracetamol if necessary. See a doctor if fever lasts more than 3 days.",
        "headache: Rest, stay hydrated, avoid bright lights. Take OTC painkillers if needed.",
        "cold and cough: Drink warm water, rest, use a humidifier. See a doctor if symptoms worsen.",
        "stomach pain: Eat light meals, drink warm water, consult a doctor if severe.",
        # Tanglish symptom-advice pairs
        "ennaikku fever irukku: Rest pannunga, thanni kudunga, paracetamol kudikalam. Fever 3 naal mela irundha doctor kitta poonga.",
        "ennaikku headache irukku: Rest pannunga, thanni kudunga, bright light avoid pannunga. OTC painkiller edunga.",
        "kodu cold um cough um irukku: Warm water kudunga, rest pannunga, humidifier use pannunga. Symptoms adhigam irundha doctor kitta poonga.",
        "stomach pain: Heavy meals avoid pannunga, warm water kudunga, severe aana doctor kitta poonga."
    ]
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local("faiss_index")

# -----------------------------
# Limited Conversation Memory
# -----------------------------
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,  # keep only last 5 messages for coherent conversation
    return_messages=True
)

# -----------------------------
# Strict Prompt: Always Answer
# -----------------------------
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and empathetic health assistant.
- Respond in the same language/style as the user (Tanglish or English).
- Give clear, concise, actionable health advice.
- IGNORE unknown words; do not try to define or interpret them literally.
- NEVER ask the user questions back.
- Use context only if relevant; do not repeat old messages.
- Focus on the latest user message.
- Avoid repeating sentences.
- If unsure, give general helpful advice for common symptoms.

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
