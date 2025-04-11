import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ctransformers

# Charger le modèle d'embedding
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Charger les textes et metadata
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)

with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# === PARTIE À EXÉCUTER UNE SEULE FOIS POUR GÉNÉRER LES EMBEDDINGS & INDEX ===
if not os.path.exists("embeddings.npy") or not os.path.exists("faiss_index.index"):
    print("Génération des embeddings et de l'index FAISS...")
    embeddings = embedding_model.encode(texts)
    np.save("embeddings.npy", embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(index, "faiss_index.index")
else:
    print("Embeddings et index FAISS déjà existants. Chargement...")