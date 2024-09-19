from flask import Flask, request, jsonify
from sent_analys import analyze_sentiment

app = Flask(__name__)


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('review_text', '')

        if not text:
            return jsonify({"error": "No review text provided"}), 400

        sentiment_result = analyze_sentiment(text)

        return jsonify({
            "review": text,
            "sentiment": sentiment_result["sentiment"],
            "confidence": sentiment_result["confidence"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
