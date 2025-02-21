#     ********************************************************************************     preprocessing ************************************************************************************************
import fitz

def count_pdf_pages(pdf_path):
    with fitz.open(pdf_path) as doc:
        return len(doc)

import os
from hotpdf import HotPdf


def extract_text_from_pdf(pdf_file_path, output_path):
    """
    Extracts text from all pages of a given PDF file and saves it to file1.txt.
    
    Parameters:
    pdf_file_path (str): The path to the input PDF file.
    output_path (str): The directory where file1.txt will be saved.
    
    Returns:
    str: The full path of the saved text file.
    """
    try:
        # Load the PDF document
        hotpdf_document = HotPdf(pdf_file_path)

        # Extract text from all pages
        full_text = ""
        for page_num in range(count_pdf_pages(pdf_file_path)):
            full_text += hotpdf_document.extract_page_text(page=page_num) + "\n\n"

        # Define the full path for the output file
        output_file = os.path.join(output_path, "file1.txt")

        # Write extracted text to the file
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(full_text)

        print(f"Text successfully saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


extract_text_from_pdf('/content/RL.pdf','/content')

final_data = [] # Dictionnaire pour stocker les sections

romans = [
    'I', 'V', 'X'
]

with open("/content/file1.txt", "r") as file:
    lines = file.readlines()

str1 = ""  # Clé (titre)
str2 = ""  # Valeur (contenu)
map2 = {}
name_root = "file1"
for line in lines:
    line = line.strip()  # Supprime les espaces et sauts de ligne

    if not line:  # Ignore les lignes vides
        continue

    first_word = line.split()[0]  # Premier mot de la ligne
    tst=0
    for x in first_word :
        if(x.isalpha() and x not in romans) :
          tst+=1
    # Vérifie si la ligne commence par un nombre romain
    if first_word[0] in romans and tst==0 :
        if(first_word[-1]=='.') :
          first_word = first_word[:-1]
        map2[first_word] = line
        if str1:  # Sauvegarde l'ancienne section avant d'en commencer une nouvelle
            chaine = str2.strip()
            name_of_part = str1
            name_of_super_part = ""
            list_of_name_of_part = str1.split(' ')
            count=0
            for r in list_of_name_of_part[0] :
              if(r=='.') : 
                count+=1
              if(count==name_of_part.count('.') ) :
                break
              name_of_super_part+=r
            #print(name_of_super_part)
            if(len(name_of_super_part)==0 or name_of_super_part not in map2.keys()) :
              final_data.append({"text": chaine,"metadata" : {"topic" : name_root, "subtopic" : name_of_part}})
            else :
              final_data.append({"text": chaine,"metadata" : {"topic" : map2[name_of_super_part], "subtopic" : name_of_part}})
        str1 = line  # Nouveau titre
        str2 = ""  # Réinitialise le contenu
    else:
        str2 += line + "\n"  # Ajoute la ligne au contenu

# Ajouter la dernière section après la boucle
if str1:
            chaine = str2.strip()
            name_of_part = str1
            name_of_super_part = ""
            count=0
            for r in name_of_part :
              if(count==name_of_part.count('.') ) :
                count+=1
                break
              name_of_super_part+=r
            if(len(name_of_super_part)==0 or name_of_super_part not in map2.keys()) :
              final_data.append({"text": chaine,"metadata" : {"topic" : name_root , "subtopic" : name_of_part}})
            else :
              print(name_of_super_part)
              final_data.append({"text": chaine,"metadata" : {"topic" : map2[name_of_super_part], "subtopic" : name_of_part}})

from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model (a small, efficient one)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
course_content = final_data
# Convert text chunks into embeddings
texts = [doc["text"] for doc in course_content]
embeddings = embedding_model.encode(texts)

# Store metadata alongside embeddings
metadata = [doc["metadata"] for doc in course_content]


import faiss

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to FAISS
index.add(np.array(embeddings))

# Save the FAISS index (optional)
faiss.write_index(index, "course_faiss.index")


def retrieve_relevant_texts(query, k=2):  # Retrieve top-k results
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(query_embedding, k)

    retrieved_texts = [texts[i] for i in indices[0]]
    return retrieved_texts

# Example query




from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import re


model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline("question-answering", model=model_name)
context =retrieved_docs[0]

result = qa_pipeline(question=query, context=context)
print(result["answer"])  # Output: "a star"


def get_answer(question, context, max_length=50, threshold=0.3):
    result = qa_pipeline(question=question, context=context, max_answer_len=max_length)

    # Customizing the response

    return {
        "answer": result["answer"],
        "confidence": round(result["score"], 2),
        "start": result["start"],
        "end": result["end"]
    }


response = get_answer(query, context, max_length=50, threshold=0.4)
print(response)





from huggingface_hub import InferenceClient
from hotpdf import HotPdf
 
# Directly pass the Hugging Face token to the InferenceClient constructor
client = InferenceClient(
    "microsoft/Phi-3.5-mini-instruct",
    token="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"  # Replace with your actual token
)

 
# Respond function for analyzing the job description
def respond_to_job_description(
    message,
    system_message,
    max_tokens,
    temperature,
    top_p,
    query,
):
    # Define the prompt for structured data extraction
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
query = str(input())
retrieved_docs = retrieve_relevant_texts(query)
#print(retrieved_docs)
print(respond_to_job_description(retrieved_docs[0], "You are a friendly assistant.", 512, 0.4, 0.9,query))


