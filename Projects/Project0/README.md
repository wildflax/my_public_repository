# Проект 3. Соревнование на Kaggle

## Оглавление  
[1. Описание проекта](https://github.com/wildflax/my_public_repository/tree/main/Projects/Project0/README.md#Описание-проекта)  
[2. Какую задачу решаем?](https://github.com/wildflax/my_public_repository/tree/main/Projects/Project0/README.md#Какую-задачу-решаем)  
[3. Краткая информация о данных](https://github.com/wildflax/my_public_repository/tree/main/Projects/Project0/README.md#Краткая-информация-о-данных)  
[4. Этапы работы над проектом](https://github.com/wildflax/my_public_repository/tree/main/Projects/Project0/README.md#Этапы-работы-над-проектом)  
[5. Результат](https://github.com/wildflax/my_public_repository/tree/main/Projects/Project0/README.md#Результат)    
[6. Выводы](https://github.com/wildflax/my_public_repository/tree/main/Projects/Project0/README.md#Выводы) 

### Описание проекта    
Создание своей первой модели, основанной на алгоритмах машинного обучения. Соревнования на Kaggle.

:arrow_up:[к оглавлению](https://github.com/wildflax/my_public_repository/tree/main/Projects/Project0/README.md#Оглавление)


### Какую задачу решаем?    
Код для создания модели предоставлен обучающей платформой и основной задачей в данном проекте является приведение данных в нужный вид.

Представьте, что вы работаете дата-сайентистом в компании Booking. Одна из проблем компании — это нечестные отели, которые накручивают себе рейтинг. Одним из способов обнаружения таких отелей является построение модели, которая предсказывает рейтинг отеля. Если предсказания модели сильно отличаются от фактического результата, то, возможно, отель ведёт себя нечестно, и его стоит проверить.

**Что практикуем**     
* Навыки для участия в соревновании на платформе Kaggle.
* Очистка данных.
* Визуализируем данные.
* Создание новых признаков.
* Отбор признаков.
* Работа с SentimentIntensityAnalyzer().


### Краткая информация о данных
* hotel_address — адрес отеля;
* review_date — дата, когда рецензент разместил соответствующий отзыв;
* average_score — средний балл отеля, рассчитанный на основе последнего комментария за последний год;
* hotel_name — название отеля;
* reviewer_nationality — страна рецензента;
* negative_review — отрицательный отзыв, который рецензент дал отелю;
* review_total_negative_word_counts — общее количество слов в отрицательном отзыв;
* positive_review — положительный отзыв, который рецензент дал отелю;
* review_total_positive_word_counts — общее количество слов в положительном отзыве;
* reviewer_score — оценка, которую рецензент поставил отелю на основе своего опыта;
* total_number_of_reviews_reviewer_has_given — количество отзывов, которые рецензенты дали в прошлом;
* total_number_of_reviews — общее количество действительных отзывов об отеле;
* tags — теги, которые рецензент дал отелю;
* days_since_review — количество дней между датой проверки и датой очистки;
* additional_number_of_scoring — есть также некоторые гости, которые просто поставили оценку сервису, но не оставили отзыв. Это число указывает, сколько там действительных оценок без проверки.
* lat — географическая широта отеля;
* lng — географическая долгота отеля.
  
:arrow_up:[к оглавлению](https://github.com/wildflax/my_public_repository/tree/main/Projects/Project0/README.md#Оглавление)


### Этапы работы над проектом  
1. Загрузка данных и первичное исследование
2. Очистка данных
3. Разведывательный анализ данных
4. Отбор признаков
5. Построение модели

:arrow_up:[к оглавлению](https://github.com/wildflax/my_public_repository/tree/main/Projects/Project0/README.md#Оглавление)


### Результаты:  
MAPE ~ 12%, наиболее важными признаками стали новые признаки сделанные из столбца с тэгами с помощью библиотеки SentimentIntensityAnalyzer().

:arrow_up:[к оглавлению](https://github.com/wildflax/my_public_repository/tree/main/Projects/Project0/README.md#Оглавление)


### Выводы:  
Самыми интересными моментами в этом проекте были создание новых признаков и участие в соревнованиях на платформе Kaggle. Среди более 500 участников соревнований у меня 48 место. Считаю прекрасным результатом, потому что достигла его исключительно благодаря обработке признаков, не изменяя алгоритм модели, на некоторых других алгоритмах(CatBoost, например) хорошие результаты получаются без глубокого разведывательного анализа.

:arrow_up:[к оглавлению](https://github.com/wildflax/my_public_repository/tree/main/Projects/Project0/README.md#Оглавление)