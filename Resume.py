import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def Resume_pipe(context):
    model_name = 'facebook/bart-large-cnn'
    api_token = "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"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_token)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=api_token)
    except Exception as e:
        print("Error loading model/tokenizer:", e)
        raise

    inputs = tokenizer.encode("summarize: " + context, return_tensors="pt", max_length=4096, truncation=False)

    summary_ids = model.generate(inputs, max_length=750, min_length=50, length_penalty=2.0, num_beams=10, early_stopping=False)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    match = re.search(r'(Summarize:\s*1\s*:?.*)',summary, re.IGNORECASE | re.DOTALL)
    if match:
           summary = match.group(1)
    return summary
def Resume_data(final_data):
  resume = []
  for x in final_data :
    resume.append(Resume_pipe(x['text']))
  return resume
