from flask import Flask, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
import pandas as pd
import traceback

app = Flask(__name__)

# ===== 1. Load dataset and create embeddings =====
df = pd.read_csv("data/health_tanglish_elaborated.csv")
symptoms = df["symptoms"].astype(str).tolist()
solutions = df["solution"].astype(str).tolist()

# Create text pairs (symptom + solution)
docs = [f"Symptom: {s}\nSolution: {sol}" for s, sol in zip(symptoms, solutions)]

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(docs, embeddings)

# ===== 2. Setup Memory =====
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ===== 3. Define LLM (HuggingFace Endpoint) =====
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-xl",   # or another hosted model
    task="text2text-generation",
    temperature=0.2,
    max_new_tokens=256
)


# ===== 4. Conversational Retrieval Chain =====
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("message", "").strip()

    if not query:
        return jsonify({"reply": "Please enter a symptom or question."})

    try:
        # Call QA chain
        result = qa.invoke({"question": query})
        return jsonify({"reply": result.get("answer", "No answer generated.")})
    except Exception as e:
        # Print detailed traceback in backend logs
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print("❌ Backend error:\n", error_msg)

        # Ensure frontend always sees *something*
        clean_msg = str(e) if str(e).strip() else "Unknown error (check backend logs)."
        return jsonify({"reply": f"⚠️ Backend error: {clean_msg}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
