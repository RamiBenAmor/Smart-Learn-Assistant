import re
import random
from sentence_transformers import SentenceTransformer
import ctransformers 
from dataProcess import structure_data
def get_first_512_words(text):
    # Utilisation d'une expression régulière pour séparer les mots et ponctuations
    words = re.findall(r'\S+|[.,!?;]', text)  # \S+ capture les mots, et [.,!?;] capture les ponctuations
    return ' '.join(words[:512])  # Récupérer les 512 premiers mots (y compris ponctuation)

def generate_quiz_from_data(final_data,n):
    context=[]
    random_numbers = random.sample(range(len(final_data)), n)
    for i in range(n) :
      context.append(get_first_512_words(final_data[random_numbers[i]]['text']))  
      #context.append(final_data[random_numbers[i]]['text'])
    quiz=[]
    for i in range(n) :
      quiz.append(generate_quiz_from_context(context[i]))
    return quiz
    
def generate_quiz_from_context(context):
    llm = ctransformers.AutoModelForCausalLM.from_pretrained(
    'C:\\Users\\ramib\\Downloads\\chatbot_interface1\\chatbot_interface\\llama-2-7b-chat.Q4_K_M.gguf',
    model_type='llama',
    max_new_tokens=256,
    temperature=0.5, #0.1 kenit
    gpu_layers=25,
    context_length=2048  
)
    prompt = f"""<s>[INST] <<SYS>>
Tu es un générateur expert de QCM pédagogiques en français. Respecte ces règles strictes :

1. **Sortie** :
   - Exactement 2 questions QCM (ni plus ni moins)
   - 4 options par question (A) à D))
   - Format standard :
     Q1. [Question]
     A) [Option]
     B) [Option]
     C) [Option]
     D) [Option]

     Q2. [Question]
     A) [Option]
     B) [Option]
     C) [Option]
     D) [Option]

2. **Taxonomie de Bloom adaptative** :
   - Analyse le contexte pour choisir le niveau cognitif dominant :
     • Connaissance (faits basiques) si le contexte est simple
     • Application (mise en pratique) si le contexte est technique
     • Analyse/Évaluation si le contexte est complexe
   - Utilise des verbes correspondants (ex: "nommer"→niveau bas, "résoudre"→niveau intermédiaire, "critiquer"→niveau avancé)

3. **Exigences** :
   - Les 2 questions doivent couvrir des aspects complémentaires du contexte
   - Option correcte aléatoirement répartie (A/B/C/D)
   - Distracteurs plausibles et thématiquement cohérents
   - Zéro anglicisme, aucune métadonnée
   - Difficulté croissante (Q1 plus simple que Q2)

Contexte : 
{context}

<</SYS>>

Génère immédiatement les 2 questions QCM sans commentaires. [/INST]"""


    try:
        # Génération contrôlée
        response = llm(
            prompt,
            max_new_tokens=256,
            temperature=0.5,  # Réduit les hallucinations
            top_p=0.95,        # Contrôle de la diversité
            stop=["</s>", "[INST]"]  # Tokens d'arrêt
        )

        # Nettoyage de la réponse
        response = response.split("[/INST]")[-1].strip()
        response = response.replace("<s>", "").replace("</s>", "").strip()

        return response

    except Exception as e:
        return f"Erreur de génération : {str(e)}"


def generate_quiz_from_data1(final_data, n, level_of_quiz):
    context = []
    random_numbers = random.sample(range(len(final_data)), n)

    for i in range(n):
        context.append(get_first_512_words(final_data[random_numbers[i]]['text']))

    quiz = []
    for i in range(n):
        quiz.append(generate_quiz_from_context1(context[i], level_of_quiz))

    return quiz

def generate_quiz_from_context1(context, level_of_quiz):
    llm = ctransformers.AutoModelForCausalLM.from_pretrained(
        'C:\\Users\\ramib\\Downloads\\chatbot_interface1\\chatbot_interface\\llama-2-7b-chat.Q4_K_M.gguf',
        model_type='llama',
        max_new_tokens=256,
        temperature=0.5,
        gpu_layers=25,
        context_length=2048
    )

    # Choix du niveau de difficulté
    if level_of_quiz == 2:
        difficulty = "moyen"
    elif level_of_quiz > 2:
        difficulty = "difficile"
    else:
        difficulty = "facile"

    prompt = f"""<s>[INST] <<SYS>>
Vous êtes un générateur de quiz en français. Suivez ces instructions à la lettre :
1. Utilisez uniquement la langue française.
2. Produisez exactement 2 questions à choix multiples.
3. Chaque question doit comporter 4 options libellées A), B), C) et D).
4. Le quiz doit être de niveau **{difficulty}**.
5. Ne préfixez pas par une introduction ni ne terminez par une conclusion.
6. N’utilisez jamais un mot ou une expression anglaise (ex. “Here”, “Great”, etc.).
7. Fournissez uniquement les questions et leurs propositions de réponses, sans autre texte.

Contexte :
{context}
<</SYS>>
[/INST]"""

    try:
        response = llm(
            prompt,
            max_new_tokens=256,
            temperature=0.5,
            top_p=0.95,
            stop=["</s>", "[INST]"]
        )
        response = response.split("[/INST]")[-1].strip()
        response = response.replace("<s>", "").replace("</s>", "").strip()
        return response
    except Exception as e:
        return f"Erreur de génération : {str(e)}"