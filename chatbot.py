import os
import pandas as pd
import nltk
import numpy as np
import joblib
import json
import csv
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

# Load necessary files
BASE_PATH = r"C:\Users\Praneetha\Downloads\Disease-Symptom-Prediction-Chatbot-main\Disease-Symptom-Prediction-Chatbot-main\Medical_dataset"

def load_json(file_name):
    with open(os.path.join(BASE_PATH, file_name), 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv(file_name):
    return pd.read_csv(os.path.join(BASE_PATH, file_name))

def load_knn_model():
    model_path = os.path.join(BASE_PATH, "..", "model", "knn.pkl")
    return joblib.load(model_path)

# Load data
intents = load_json("intents_short.json")
df_tfidf = load_csv("tfidfsymptoms.csv")
df_train = load_csv("Training.csv")
knn = load_knn_model()

vocab = list(df_tfidf.columns)
disease_labels = df_train.iloc[:, -1].to_list()
all_symptoms = [col.replace('_', ' ') for col in df_train.columns[:-1]]

# Helper functions
def preprocess_sent(sent):
    tokens = nltk.word_tokenize(sent)
    return ' '.join([lemmatizer.lemmatize(w.lower()) for w in tokens if w.isalpha() and w not in stopwords.words('english')])

def bag_of_words(tokenized_sentence, all_words):
    return np.array([1.0 if w in tokenized_sentence else 0.0 for w in all_words], dtype=np.float32)

def predict_symptom(symptom):
    symptom = preprocess_sent(symptom)
    bow_vector = bag_of_words(symptom, vocab)
    res = cosine_similarity([bow_vector], df_tfidf).flatten()
    sorted_indices = np.argsort(res)[::-1]
    return vocab[sorted_indices[0]] if res[sorted_indices[0]] > 0 else None

def one_hot_encode(symptoms):
    encoded = np.zeros((1, len(all_symptoms)))
    for symptom in symptoms:
        if symptom in all_symptoms:
            encoded[0, all_symptoms.index(symptom)] = 1
    return pd.DataFrame(encoded, columns=all_symptoms)

def get_disease_prediction(symptoms):
    encoded_symptoms = one_hot_encode(symptoms)
    encoded_symptoms_array = encoded_symptoms.to_numpy()  # Convert DataFrame to NumPy array
    return knn.predict(encoded_symptoms_array)[0]


def chatbot():
    print("Welcome to the Disease Prediction Chatbot!")
    name = input("Enter your name: ")
    print(f"Hello {name}, please describe your symptoms.")
    
    symptoms = []
    for _ in range(2):
        symptom = input("Enter a symptom: ")
        predicted_symptom = predict_symptom(symptom)
        if predicted_symptom:
            symptoms.append(predicted_symptom)
    
    disease = get_disease_prediction(symptoms)
    print(f"Based on your symptoms, you may have: {disease}")

if __name__ == "__main__":
    chatbot()
