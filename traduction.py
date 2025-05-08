from transformers import MarianMTModel, MarianTokenizer
# Variables globales pour mémoriser les objets chargés
model = None
tokenizer = None

def traduction(text):
    global model, tokenizer  # permet de modifier les variables globales

    if model is None or tokenizer is None:
        print("Chargement du modèle...")
        model_name = "Helsinki-NLP/opus-mt-en-fr"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)