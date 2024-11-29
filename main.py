from flask import Flask, request, jsonify
import json
import nlp_utils 

app = Flask(__name__)

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    processed_words = nlp_utils.preprocess_text(text)
    return jsonify({'processed_words': processed_words})

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    top_results = nlp_utils.find_top_n_relevant_documents(query, documents["documents"], vectorizer, tfidf_matrix)
    return jsonify({'top_results': top_results})

if __name__ == '__main__':
    with open("documents.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    vectorizer, tfidf_matrix = nlp_utils.create_index(documents["documents"])

    app.run(debug=True)