# Тестовое задание
## Описание задания
### Middle Python Developer
#### Часть 1: NLP-пайплайн
#### Часть 2: Поиск по тексту (упрощенный RAG)
Реализовать базовый NLP-пайплайн для обработки текстовых данных.
* Токенизация текста.
* Очистка текста от стоп-слов.
* Приведение к нижнему регистру и лемматизация.
* Создайте индекс из текстов в базе данных, используя модель TF-IDF.
* Реализуйте функцию, которая принимает на вход текст запроса и возвращает
топ-3 наиболее релевантных текста из базы. 
* Примените функцию в API. 
* Настройте систему поиска по тексту (по принципу RAG или упрощенной версии).
Представьте, что компания разрабатывает бота, который должен распознавать запросы пользователей на русском языке и находить нужную информацию в базе данных.

Можно использовать библиотеки nltk или spaCy.

Для этого нужно:
* Реализуйте упрощенный поиск по тексту, чтобы по запросу пользователя находить наиболее релевантные результаты из заранее подготовленного текстового файла (например, база отзывов на товары).
* RAG API, которая на вход принимает запрос и возвращает список из 3 наиболее релевантных результатов (приложить скрипт обращения к API). 
* Напишите на Python пайплайн для обработки текстовых запросов.
* Пайплайн должен включать следующие этапы:
1. Токенизация текста
2. Очистка текста от стоп-слов
3. Приведение к нижнему регистру и лемматизация
4. Создание индекса из текстов в базе данных, используя модель TF-IDF
5. Реализация функции для поиска наиболее релевантных текстов
6. Настройка системы поиска по тексту (по принципу RAG или упрощенной версии)
	7. Валидация пайплана с использованием тестовых данных 
8. Отправка результата API
9. Обработка ошибок и исключений при выполнении запроса
10. Написание документации по API и пайплайну