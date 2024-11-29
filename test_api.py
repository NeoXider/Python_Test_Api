import json
from nlp_utils import preprocess_text, create_index, find_top_n_relevant_documents, print_debug_info, print_tfidf_vectors

# Загрузка документов из файла (пример)
with open("documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

vectorizer, tfidf_matrix = create_index(documents["documents"])

if __name__ == '__main__':
    test_query = "хороший товар"

    expected_result = {
        'top_results': ['Хороший товар, быстрая доставка.',
                        'Хорошее обслуживание, товар в отличном состоянии.',
                        'Хороший товар, быстрая доставка. Отличный выбор!']
    }

    try:
        top_results = find_top_n_relevant_documents(
            test_query, documents["documents"], vectorizer, tfidf_matrix)
        result = {'top_results': top_results}

        print_debug_info(documents["documents"], vectorizer, test_query)
        print_tfidf_vectors(documents["documents"], vectorizer)

        print("\nВопрос:", test_query)
        print("Отправлено:", preprocess_text(test_query))
        print("Должно быть:", expected_result['top_results'])
        print("Результат:", result['top_results'])

        assert result == expected_result, f"Ожидаемый результат: {expected_result}\n, но получил: {result}"

    except Exception as e:
        print(f"Произошла ошибка при тестовом поиске: {e}")
