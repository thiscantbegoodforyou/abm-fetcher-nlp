from flask import Flask, request, jsonify
import requests
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

CATEGORIES = ["Hiring", "Funding", "Expansion", "Partnership", "Compliance"]
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # stored in Render environment

@app.route('/classify', methods=['POST'])
def classify_article():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Missing 'text' in request"}), 400

    response = requests.post(
        "https://api-inference.huggingface.co/models/facebook/bart-large-mnli",
        headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
        json={"inputs": text, "parameters": {"candidate_labels": CATEGORIES}}
    )

    if response.status_code != 200:
        return jsonify({"error": "HuggingFace API request failed"}), 500

    result = response.json()
    top_label = result['labels'][0]
    confidence = result['scores'][0]

    return jsonify({
        "category": top_label,
        "confidence": round(confidence, 3),
        "all_scores": dict(zip(result['labels'], map(lambda x: round(x, 3), result['scores'])))
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
