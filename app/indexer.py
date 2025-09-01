from __future__ import annotations
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .nlp import embed_texts

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "health_tanglish.csv")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "storage", "symptom_index.json")

def build_index() -> Dict[str, Any]:
    try:
        print("📂 Loading CSV from:", DATA_PATH)
        df = pd.read_csv(DATA_PATH)
        print("✅ CSV Columns:", df.columns.tolist())
        print("✅ Number of rows:", len(df))

        # Pick the right symptom column
        if "symptom_text" in df.columns:
            text_col = "symptom_text"
        elif "symptoms" in df.columns:
            text_col = "symptoms"
        else:
            print("❌ No valid symptom column found")
            return {"symptoms": [], "solutions": [], "embeddings": []}

        if "solution" not in df.columns:
            print("❌ No 'solution' column found")
            return {"symptoms": [], "solutions": [], "embeddings": []}

        texts = df[text_col].astype(str).fillna("").tolist()
        sols = df["solution"].astype(str).fillna("").tolist()

        # Generate embeddings
        if not texts:
            print("⚠️ No symptom texts found")
            return {"symptoms": [], "solutions": [], "embeddings": []}

        embs = embed_texts(texts)
        print("✅ Generated embeddings:", len(embs))

        # Save index
        data = {
            "symptoms": texts,
            "solutions": sols,
            "embeddings": np.asarray(embs, dtype=np.float32).tolist(),
        }
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print("💾 Index saved at:", INDEX_PATH)
        return data

    except Exception as e:
        print("❌ Error in build_index:", e)
        return {"symptoms": [], "solutions": [], "embeddings": []}

def load_index() -> Dict[str, Any]:
    try:
        if not os.path.exists(INDEX_PATH):
            print("⚠️ Index not found, rebuilding...")
            return build_index()
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("✅ Index loaded with", len(data.get("symptoms", [])), "symptoms")
        return data
    except Exception as e:
        print("❌ Error in load_index:", e)
        return {"symptoms": [], "solutions": [], "embeddings": []}

def search_symptom(query: str, top_k: int = 3) -> Tuple[int, str, float]:
    """Return (best_idx, solution, score) via cosine similarity."""
    idx = load_index()

    if not idx["symptoms"]:
        return -1, "No data available", 0.0

    try:
        q = np.array(embed_texts([query])[0], dtype=np.float32)
        M = np.array(idx["embeddings"], dtype=np.float32)

        if M.shape[0] == 0:
            return -1, "No data available", 0.0

        sims = M @ q
        best = int(np.argmax(sims))
        return best, idx["solutions"][best], float(sims[best])

    except Exception as e:
        print("❌ Error in search_symptom:", e)
        return -1, "Error while searching", 0.0
