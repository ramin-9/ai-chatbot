from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_processing import load_faq_data, preprocess_text

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

faqs = load_faq_data()
questions = [preprocess_text(item["question"]) for item in faqs]
answers = [item["answer"] for item in faqs]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_best_response(user_input):
    user_input = preprocess_text(user_input)
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_idx = np.argmax(similarities)

    if similarities[0, best_match_idx] < 0.2:
        return "Sorry, I don't understand. Can you rephrase?"
    return answers[best_match_idx]

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    response = get_best_response(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
