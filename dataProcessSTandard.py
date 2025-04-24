from transformers import AutoTokenizer  # Pour tokeniser le texte

def split_document_by_tokens(file_p, max_tokens=500, overlap=50, model_name="NousResearch/Llama-2-7b-chat-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    with open(file_p, "r", encoding="utf-8") as file:
        text = file.read()

    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        chunks.append({
            "text": chunk_text,
            "metadata": {"topic": ".", "subtopic": "."}
        })

        start += max_tokens - overlap
    return chunks
