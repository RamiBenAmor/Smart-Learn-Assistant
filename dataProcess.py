#import fitz
import pdfplumber
import os
import nltk
import json
import pickle
from nltk.tokenize import word_tokenize
nltk.download('punkt')
def extract_text_from_pdf(pdf_file_path, output_path):
    """
    Extracts text from all pages of a given PDF file using pdfplumber,
    and saves it to file1.txt.

    Parameters:
    pdf_file_path (str): The path to the input PDF file.
    output_path (str): The directory where file1.txt will be saved.

    Returns:
    str: The full path of the saved text file.
    """
    try:
        full_text = ""
        with pdfplumber.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"

        output_file = os.path.join(output_path, "file1.txt")
        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write(full_text)

        print(f"Text successfully saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"An error occurred: {e}")
        return None





def structure_data(file_p):

  final_data = [] # Dictionnaire pour stocker les sections

  romans = [
      'I', 'V', 'X'
  ]

  with open(file_p, "r",encoding="utf-8") as file:
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
              # resume = resumer_cour(chaine, system_message, max_tokens, temperature, top_p)
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
  return final_data

extract_text_from_pdf('C:\\Users\\ramib\\OneDrive\\Bureau\\PCD\\RLC.pdf','C:\\Users\\ramib\\OneDrive\\Bureau\\PCD\\')




nltk.download('punkt_tab')

def count_tokens_nltk(word):
    tokens = nltk.tokenize.word_tokenize(word)
    return len(tokens)




def pre_of_the_preprocessing(final_data) :

  final_of_the_final_data =[]
  for x in final_data :
     subcontext = ""
     nbr_of_tokens = 0
     mots = x["text"].split()
     for word in mots :
       nbr_of_tokens  = nbr_of_tokens + count_tokens_nltk(word)
       if(nbr_of_tokens >512) :

         final_of_the_final_data.append({'text':subcontext,'metadata':{'topic':x['metadata']['topic'],'subtopic':x['metadata']['subtopic']}})
         subcontext=""
         nbr_of_tokens=0
       subcontext = subcontext + word +' '

     final_of_the_final_data.append({'text':subcontext,'metadata':{'topic':x['metadata']['topic'],'subtopic':x['metadata']['subtopic']}})
  return final_of_the_final_data

final_data = structure_data('C:\\Users\\ramib\\OneDrive\\Bureau\\PCD\\file1.txt')
#y=pre_of_the_preprocessing(final_data)
y=final_data
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(y, f, ensure_ascii=False, indent=4)

texts = [entry["text"] for entry in y]
metadata = [entry["metadata"] for entry in y]
# Sauvegarder les textes et métadonnées dans des fichiers .pkl
with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)

with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)




