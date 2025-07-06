from flask import Flask, request, jsonify, render_template
from Youtube_RAG import answer_query
from flask_cors import CORS

app= Flask(__name__)
CORS(app) # TO allow access from browser extensions or frontend

@app.route('/')
def welcome():
    return render_template("index.html")

@app.route('/ask', methods=["POST"])
def ask():
    data = request.get_json()
    url= data.get('url')
    question = data.get('question')

    if not url or not question:
        return jsonify({"error": "URL or question are required"}), 400

    try:
        result = answer_query(url, question)
        return jsonify({"answer": result["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)