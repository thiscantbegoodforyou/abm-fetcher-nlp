from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Optional: allows requests from WordPress plugin

# Load model once at startup
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")

CATEGORIES = ["Hiring", "Funding", "Expansion", "Partnership", "Compliance"]

@app.route('/classify', methods=['POST'])
def classify_article():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "Missing 'text' in request"}), 400

    result = classifier(text, CATEGORIES)
    top_label = result['labels'][0]
    confidence = result['scores'][0]

    return jsonify({
        "category": top_label,
        "confidence": round(confidence, 3),
        "all_scores": dict(zip(result['labels'], map(lambda x: round(x, 3), result['scores'])))
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default to 5000 locally
    app.run(host="0.0.0.0", port=port)