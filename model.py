from tensorflow.keras.models import Sequential  # Импортируем Sequential для создания последовательной модели
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout  # Импортируем необходимые слои для нейронной сети
import numpy as np  # Импортируем библиотеку numpy для работы с массивами
import re  # Импортируем библиотеку re для работы с регулярными выражениями
from sklearn.model_selection import train_test_split  # Импортируем функцию для разделения данных на тренировочные и тестовые наборы
import mailbox  # Импортируем библиотеку для работы с mbox файлами
from tensorflow.keras.preprocessing.text import Tokenizer  # Импортируем Tokenizer для токенизации текста
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Импортируем pad_sequences для дополнения последовательностей до одной длины
import matplotlib.pyplot as plt  # Импортируем matplotlib для построения графиков
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # Импортируем метрики для оценки модели

# Функция для загрузки писем из файлов mbox фишинговых и легитимных писем
def load_emails(phishing_file_path, legitimate_file_path):
    print("Загрузка фишинговых писем...")  # Печатаем сообщение о начале загрузки фишинговых писем
    phishing_mbox = mailbox.mbox(phishing_file_path)  # Загружаем mbox файл с фишинговыми письмами
    phishing_emails = []  # Создаем пустой список для фишинговых писем
    for message in phishing_mbox:  # Проходим по всем сообщениям в mbox файле
        if message.is_multipart():  # Если сообщение состоит из нескольких частей
            for part in message.walk():  # Проходим по всем частям сообщения
                if part.get_content_type() == "text/plain":  # Если тип содержимого текст
                    payload = part.get_payload(decode=True)  # Получаем содержимое части сообщения
                    if payload:  # Если содержимое не пустое
                        phishing_emails.append(payload.decode(errors='ignore'))  # Декодируем содержимое и добавляем в список фишинговых писем
        else:
            payload = message.get_payload(decode=True)  # Если сообщение не состоит из нескольких частей, получаем содержимое
            if payload:  # Если содержимое не пустое
                phishing_emails.append(payload.decode(errors='ignore'))  # Декодируем содержимое и добавляем в список фишинговых писем

    print("Загрузка легитимных писем...")  # Печатаем сообщение о начале загрузки легитимных писем
    legitimate_mbox = mailbox.mbox(legitimate_file_path)  # Загружаем mbox файл с легитимными письмами
    legitimate_emails = []  # Создаем пустой список для легитимных писем
    for message in legitimate_mbox:  # Проходим по всем сообщениям в mbox файле
        if message.is_multipart():  # Если сообщение состоит из нескольких частей
            for part in message.walk():  # Проходим по всем частям сообщения
                if part.get_content_type() == "text/plain":  # Если тип содержимого текст
                    payload = part.get_payload(decode=True)  # Получаем содержимое части сообщения
                    if payload:  # Если содержимое не пустое
                        legitimate_emails.append(payload.decode(errors='ignore'))  # Декодируем содержимое и добавляем в список легитимных писем
        else:
            payload = message.get_payload(decode=True)  # Если сообщение не состоит из нескольких частей, получаем содержимое
            if payload:  # Если содержимое не пустое
                legitimate_emails.append(payload.decode(errors='ignore'))  # Декодируем содержимое и добавляем в список легитимных писем

    return phishing_emails, legitimate_emails  # Возвращаем списки фишинговых и легитимных писем

# Функция для предобработки писем
def preprocess_emails(emails):
    print("Предобработка писем...")  # Печатаем сообщение о начале предобработки писем
    cleaned_emails = []  # Создаем пустой список для очищенных писем
    for email in emails:  # Проходим по всем письмам
        email = re.sub(r'\W', ' ', email)  # Удаляем специальные символы
        email = email.lower()  # Преобразуем текст к нижнему регистру
        cleaned_emails.append(email)  # Добавляем очищенное письмо в список
    return cleaned_emails  # Возвращаем список очищенных писем

# Загрузка и предобработка писем
print("Загрузка и предобработка писем...")  # Печатаем сообщение о начале загрузки и предобработки писем
phishing_emails, legitimate_emails = load_emails('emails-phishing.mbox', 'emails-normal.mbox')  # Загружаем фишинговые и легитимные письма
emails = phishing_emails + legitimate_emails  # Объединяем списки фишинговых и легитимных писем
cleaned_emails = preprocess_emails(emails)  # Предобрабатываем все письма

# Создание меток для датасета (1 для фишинга, 0 для легитимных)
labels = np.array([1] * len(phishing_emails) + [0] * len(legitimate_emails))  # Создаем массив меток

# Разделение датасета на тренировочные и тестовые наборы
print("Разделение данных на тренировочные и тестовые...")  # Печатаем сообщение о начале разделения данных
X_train, X_test, y_train, y_test = train_test_split(cleaned_emails, labels, test_size=0.2, random_state=42)  # Разделяем данные

# Токенизация и дополнение последовательностей
print("Токенизация и дополнение последовательностей...")  # Печатаем сообщение о начале токенизации и дополнения последовательностей
tokenizer = Tokenizer(num_words=10000)  # Создаем токенизатор
tokenizer.fit_on_texts(X_train)  # Обучаем токенизатор на тренировочных данных
X_train_seq = tokenizer.texts_to_sequences(X_train)  # Преобразуем тренировочные тексты в последовательности
X_test_seq = tokenizer.texts_to_sequences(X_test)  # Преобразуем тестовые тексты в последовательности

# Дополнение последовательностей до одной длины
X_train_pad = pad_sequences(X_train_seq, maxlen=100)  # Дополняем тренировочные последовательности до длины 100
X_test_pad = pad_sequences(X_test_seq, maxlen=100)  # Дополняем тестовые последовательности до длины 100

# Определение модели нейронной сети
print("Определение модели нейронной сети...")  # Печатаем сообщение о начале определения модели
model = Sequential([  # Создаем последовательную модель
    Embedding(input_dim=10000, output_dim=128, input_length=100),  # Слой Embedding для представления слов
    Conv1D(filters=128, kernel_size=5, activation='relu'),  # Слой 1D свертки для извлечения признаков
    MaxPooling1D(pool_size=5),  # Слой максимального пула для уменьшения размерности
    LSTM(units=128),  # LSTM слой для захвата долгосрочных зависимостей
    Dense(units=128, activation='relu'),  # Полносвязный слой с ReLU активацией
    Dropout(rate=0.5),  # Слой Dropout для регуляризации
    Dense(units=1, activation='sigmoid')  # Выходной слой для бинарной классификации
])

# Компиляция модели
print("Компиляция модели...")  # Печатаем сообщение о начале компиляции модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Компилируем модель с оптимизатором adam и функцией потерь binary_crossentropy

# Обучение модели
print("Обучение модели...")  # Печатаем сообщение о начале обучения модели
history = model.fit(X_train_pad, y_train, epochs=20, batch_size=32, validation_data=(X_test_pad, y_test))


# Plot training accuracy and loss values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпоха')

# Plot training loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Потери модели')
plt.ylabel('Потери')
plt.xlabel('Эпоха')

plt.tight_layout()
plt.show()

# Evaluate model
def evaluate_model(model, X_test_pad, y_test):
    print("Оценка модели...")
    # Make predictions
    predictions = model.predict(X_test_pad)

    # Convert predictions to binary class labels
    predicted_classes = (predictions >= 0.5).astype(int).flatten()

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, predicted_classes)
    precision = precision_score(y_test, predicted_classes)
    recall = recall_score(y_test, predicted_classes)
    f1 = f1_score(y_test, predicted_classes)
    cm = confusion_matrix(y_test, predicted_classes)

    # Print the results
    print(f"Точность модели: {accuracy:.4f}")
    print(f"Точность (Precision): {precision:.4f}")
    print(f"Полнота (Recall): {recall:.4f}")
    print(f"F1-оценка (F1 Score): {f1:.4f}")
    print(f"Матрица ошибок (Confusion Matrix):\n{cm}")

# Evaluate the model
evaluate_model(model, X_test_pad, y_test)

print("Оценка завершена.")
