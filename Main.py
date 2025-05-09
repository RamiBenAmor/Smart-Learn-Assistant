import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ctransformers
from traduction import traduction
def retrieve_relevant_texts(query, k=1):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Charger les fichiers Pickle
    with open("texts.pkl", "rb") as f:
      texts = pickle.load(f)
    index1 = faiss.read_index("faiss_index.index")
    # Convertir la requête en embedding
    query_embedding = embedding_model.encode([query])

    # Recherche dans l'index FAISS
    _, indices = index1.search(np.array(query_embedding, dtype=np.float32), k)  # Recherche dans l'index

    # Récupérer les textes pertinents en fonction des indices retournés
    retrieved_texts = [texts[i] for i in indices[0]]

    return retrieved_texts



def get_contextual_response(context,question):
    llm = ctransformers.AutoModelForCausalLM.from_pretrained(
    'C:\\Users\\ramib\\OneDrive\\Bureau\\PCD\\llama-2-7b-chat.Q4_K_M.gguf',
    model_type='llama',
    max_new_tokens=512,
    temperature=0.6, #0.1 kenit
    gpu_layers=25,
    context_length=2048  
)
    # Construction stricte du prompt LLaMA 2
    prompt = f"""<s>[INST] <<SYS>>
Vous êtes un expert IA hautement compétent, précis et détaillé. Vous fournissez toujours des réponses complètes, exactes et parfaitement structurées en anglais. Vous respectez strictement ces règles :

1. Analyse approfondie du contexte fourni
2. Réponses précises et factuelles
3. Structure claire avec paragraphes organisés
4. Langage professionnel et adapté
5. Justification des réponses quand nécessaire
6. Vérification interne de la cohérence

Contexte :
{context}
<</SYS>>

Question : {question}

Pour répondre :
1. Comprenez parfaitement la question
2. Analysez chaque élément du contexte pertinent
3. Synthétisez les informations
4. Formulez une réponse complète
5. Vérifiez l'exactitude avant de répondre

Fournissez maintenant la meilleure réponse possible : [/INST]"""
    try:
        # Génération contrôlée
        response = llm(
            prompt,
            max_new_tokens=512,
            temperature=0.6,  # Réduit les hallucinations
            top_p=0.95,        # Contrôle de la diversité
            stop=["</s>", "[INST]"]  # Tokens d'arrêt
        )

        # Nettoyage de la réponse
        response = response.split("[/INST]")[-1].strip()
        response = response.replace("<s>", "").replace("</s>", "").strip()
        response=traduction(response)
        return response

    except Exception as e:
        return f"Erreur de génération : {str(e)}"  
