import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
def generate_embeddings_and_faiss():
  embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

  with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)

  print("Génération des embeddings et de l'index FAISS...")
  embeddings = embedding_model.encode(texts)
  np.save("embeddings.npy", embeddings)

  dimension = embeddings.shape[1]
  index = faiss.IndexFlatL2(dimension)
  index.add(np.array(embeddings, dtype=np.float32))
  faiss.write_index(index, "faiss_index.index")

