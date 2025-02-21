from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Initialiser Flask
app = Flask(__name__)

# Charger le modèle de transformation de phrases
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Charger les fichiers Pickle
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)

with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Créer l'index FAISS
embeddings = embedding_model.encode(texts)
dimension = embeddings.shape[1]

# Créer un index L2 de FAISS (distance euclidienne)
index = faiss.IndexFlatL2(dimension)  # Assurez-vous d'utiliser cette ligne pour créer un objet d'index valide

# Ajouter les embeddings à l'index
index.add(np.array(embeddings, dtype=np.float32))


# Fonction pour récupérer les textes pertinents
def retrieve_relevant_texts(query, k=2):
    # Convertir la requête en embedding
    query_embedding = embedding_model.encode([query])
    
    # Recherche dans l'index FAISS
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), k)  # Recherche dans l'index
    
    # Récupérer les textes pertinents en fonction des indices retournés
    retrieved_texts = [texts[i] for i in indices[0]]
    
    return retrieved_texts


# Initialiser le client HuggingFace pour la réponse
client = InferenceClient(
    "microsoft/Phi-3.5-mini-instruct",
    token="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"  # Remplacer par ton token réel
)


# Fonction de réponse
def respond_to_job_description(message, system_message, max_tokens, temperature, top_p, query):
    # Définir le prompt pour l'extraction des données structurées
    prompt = f"""{query} : {message} 
    """
    
    messages = [{"role": "system", "content": system_message}]
    messages.append({"role": "user", "content": prompt})
    
    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
    
    return response


# Route principale
@app.route('/')
def index():
    return render_template('index.html')


# Route de chat
@app.route('/chat', methods=['POST'])
def chat():
    # Récupérer l'entrée de l'utilisateur
    user_input = request.json.get("message")
    
    # Récupérer les textes pertinents pour la question
    relevant_texts = retrieve_relevant_texts(user_input, k=2)
    
    # Utiliser les textes récupérés pour obtenir une réponse du modèle HuggingFace
    context = " ".join(relevant_texts)
    system_message = "You are a friendly assistant."
    query = user_input
    response = respond_to_job_description(context, system_message, 512, 0.4, 0.9, query)
    
    # Retourner la réponse sous forme de JSON
    return jsonify({"response": response})


# Démarrer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
