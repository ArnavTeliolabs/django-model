"""
Place your models or data processing functions here.
"""

import nltk
import string
import re
import numpy as np
from django.db import models
from transformers import BertTokenizer, TFBertModel
import faiss

# Ensure nltk downloads are run once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')

def data_preprocess(description):
    description = description.lower()
    description = re.sub(r"<[^>]+>", "", description)
    tokens = nltk.word_tokenize(description)
    filtered_tokens = [token for token in tokens if token not in string.punctuation]
    stopwords = nltk.corpus.stopwords.words("english")
    filtered_tokens = [token for token in filtered_tokens if token.lower() not in stopwords]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    tagged_tokens = nltk.pos_tag(lemmatized_tokens)
    return tagged_tokens

def get_bert_embeddings(tokens):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(tokens, return_tensors='tf', padding=True, truncation=True, is_split_into_words=True)
    outputs = model(inputs)
    pooled_output = outputs.last_hidden_state[:, 0, :]
    return pooled_output

proxy_dataset = {
    "Users are reporting issues with the application's login process...": "Check the authentication server logs...",
    "Users are experiencing delays when trying to upload files...": "Inspect the server load and bandwidth usage...",
    # Add other query-solution pairs here
}

embeddings_dict = {}
for query, solution in proxy_dataset.items():
    preprocessed_query = data_preprocess(query)
    query_embedding = get_bert_embeddings(preprocessed_query)
    embeddings_dict[query] = {
        "embedding": query_embedding,
        "solution": solution
    }

dimension = 768  # BERT base embedding dimension
index = faiss.IndexFlatL2(dimension)
embedding_vectors = []
solutions = []

for query, data in embeddings_dict.items():
    if query in proxy_dataset:
        embedding = data['embedding'].numpy()
        averaged_embedding = np.mean(embedding, axis=0, keepdims=True)
        solution = proxy_dataset[query]
        embedding_vectors.append(averaged_embedding)
        solutions.append(solution)

if embedding_vectors:
    embedding_vectors = np.vstack(embedding_vectors)
    index.add(embedding_vectors)

def preprocess_and_embed(query):
    preprocessed_query = data_preprocess(query)
    tokens = [" ".join([word for word, tag in preprocessed_query])]
    embedding = get_bert_embeddings(tokens)
    return embedding

def find_solution(query):
    preprocessed_query = data_preprocess(query)
    query_embedding = get_bert_embeddings(preprocessed_query).numpy()
    distances, indices = index.search(query_embedding, 1)
    solution = solutions[indices[0][0]]
    return solution
