import json
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt_tab")

# Load the FAQ data
def load_faq_data(file_path="faq.json"):
    with open(file_path, "r") as file:
        return json.load(file)

# Preprocess text: Tokenization
def preprocess_text(text):
    return " ".join(word_tokenize(text.lower()))

# Convert FAQ data into a training format
def prepare_training_data():
    faqs = load_faq_data()
    questions = [preprocess_text(item["question"]) for item in faqs]
    answers = [item["answer"] for item in faqs]
    return questions, answers

if __name__ == "__main__":
    questions, answers = prepare_training_data()
    print("Sample Processed Questions:", questions[:2])
