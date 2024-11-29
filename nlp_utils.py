import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("ru_core_news_sm")
debug = False

def preprocess_text(text):
    doc = nlp(text)
    
    result = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]

    if debug:
        print(f"Original: {text}")
        print(f"Lemmas: {result}")
        print(f"Processed: {' '.join(result)}")

    return " ".join(result)

def create_index(documents):
    processed_docs = [preprocess_text(doc) for doc in documents]
    
    if debug:
        print("\nPreprocessed Documents:")
        for i, doc in enumerate(processed_docs):
            print(f"{i}: {doc}")
    
    tfidf_vectorizer = TfidfVectorizer(
        stop_words=None,
        norm='l2',
        use_idf=True,
        max_df=0.85,
        min_df=0.15
    )
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)
    
    return tfidf_vectorizer, tfidf_matrix

def find_top_n_relevant_documents(query, documents, vectorizer, matrix, n=3):
    processed_query = preprocess_text(query)
    query_vec = vectorizer.transform([processed_query])
    
    cosine_similarities = cosine_similarity(query_vec, matrix).flatten()
    top_n_indices = cosine_similarities.argsort()[::-1][:n]
    
    return [documents[i] for i in top_n_indices]

def print_debug_info(documents, vectorizer, query):
    processed_docs = [preprocess_text(doc) for doc in documents]
    tfidf_vectorizer = vectorizer
    tfidf_matrix = tfidf_vectorizer.transform(processed_docs)
    
    query_vec = tfidf_vectorizer.transform([query])
    
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_n_indices = cosine_similarities.argsort()[::-1]
    
    print("\nDebug Information:")
    for i, sim in zip(top_n_indices, cosine_similarities[top_n_indices]):
        print(f"Document: {documents[i]}, Score: {sim:.4f}")

def print_tfidf_vectors(documents, vectorizer):
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    feature_names = vectorizer.get_feature_names_out()
    for i in range(len(documents)):
        doc_vec = tfidf_matrix[i]
        scores = [(feature_names[idx], round(doc_vec[0, idx], 4)) 
                  for idx in doc_vec.indices if doc_vec[0, idx] > 0]
        print(f"\nDocument {i}: {documents[i]}")
        for term, score in sorted(scores, key=lambda x: -x[1]):
            print(f"  {term}: {score:.4f}")