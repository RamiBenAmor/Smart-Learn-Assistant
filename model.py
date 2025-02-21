from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import re

def read_file_to_dict(file_path):
    # Read the entire file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regex pattern to capture key and value.
    # re.DOTALL ensures that '.' matches newline characters as well.
    pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'

    # Find all matches in the content
    matches = re.findall(pattern, content, re.DOTALL)

    # Convert the list of tuples into a dictionary
    dictionary = {key.strip(): value.strip() for key, value in matches}
    return dictionary

# Usage example:
file_path = '/content/data_model.txt'
result_dict = read_file_to_dict(file_path)
#print(result_dict.keys())


titles = list(result_dict.keys())
# Question to match
question = "what's a model finetuning ?"

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
corpus = [question] + titles  # Combine question and titles
X = vectorizer.fit_transform(corpus)

# Compute cosine similarity between question and all titles
similarities = cosine_similarity(X[0], X[1:])  # Compare question (X[0]) with titles (X[1:])

# Find the most relevant title
best_match_index = similarities.argmax()
most_relevant_title = titles[best_match_index]
model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline("question-answering", model=model_name)
context = result_dict[most_relevant_title]

result = qa_pipeline(question=question, context=context)
print(result["answer"])  # Output: "a star"

print(most_relevant_title)
