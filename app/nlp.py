
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

_nlp = None
_embedder = None

def get_spacy():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    return np.array(model.encode(texts, normalize_embeddings=True))

def extract_entities(text: str):
    nlp = get_spacy()
    doc = nlp(text)
    return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]

TANGLISH_HINTS = [
    "irukku","enna","enaku","ennaku","ennakku","vali","suthu","thala","satham",
    "sogam","pasikkuthu","kudinga","tanni","vayiru","vanti","thookam",
    "naale","innaiku","rendu","kadi","udambu","sarumam"
]
def is_tanglish(text: str) -> bool:
    t = text.lower()
    return any(tok in t for tok in TANGLISH_HINTS)
