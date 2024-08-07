from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
import numpy as np
import re
from sklearn.model_selection import train_test_split
import mailbox
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load emails from both phishing and legitimate email mbox files
def load_emails(phishing_file_path, legitimate_file_path):
    print("Загрузка фишинговых писем...")
    phishing_mbox = mailbox.mbox(phishing_file_path)
    phishing_emails = []
    for message in phishing_mbox:
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        phishing_emails.append(payload.decode(errors='ignore'))
        else:
            payload = message.get_payload(decode=True)
            if payload:
                phishing_emails.append(payload.decode(errors='ignore'))

    print("Загрузка легитимных писем...")
    legitimate_mbox = mailbox.mbox(legitimate_file_path)
    legitimate_emails = []
    for message in legitimate_mbox:
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        legitimate_emails.append(payload.decode(errors='ignore'))
        else:
            payload = message.get_payload(decode=True)
            if payload:
                legitimate_emails.append(payload.decode(errors='ignore'))

    return phishing_emails, legitimate_emails

# Preprocess the emails
def preprocess_emails(emails):
    print("Предобработка писем...")
    cleaned_emails = []
    for email in emails:
        email = re.sub(r'\W', ' ', email)  # Remove special characters
        email = email.lower()  # Convert to lowercase
        cleaned_emails.append(email)
    return cleaned_emails

# Load and preprocess the emails
print("Загрузка и предобработка писем...")
phishing_emails, legitimate_emails = load_emails('emails-phishing.mbox', 'emails-normal.mbox')
emails = phishing_emails + legitimate_emails
cleaned_emails = preprocess_emails(emails)

# Create labels for the dataset (1 for phishing, 0 for non-phishing)
labels = np.array([1] * len(phishing_emails) + [0] * len(legitimate_emails))

# Split the dataset into training and testing sets
print("Разделение данных на тренировочные и тестовые...")
X_train, X_test, y_train, y_test = train_test_split(cleaned_emails, labels, test_size=0.2, random_state=42)

# Tokenization and padding
print("Токенизация и дополнение последовательностей...")
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform input size
X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

# Define the neural network model
print("Определение модели нейронной сети...")
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),  # Embedding layer for word representation
    Conv1D(filters=128, kernel_size=5, activation='relu'),  # 1D convolutional layer for feature extraction
    MaxPooling1D(pool_size=5),  # Max pooling layer to reduce dimensionality
    LSTM(units=128),  # LSTM layer for capturing long-term dependencies
    Dense(units=128, activation='relu'),  # Fully connected layer with ReLU activation
    Dropout(rate=0.5),  # Dropout layer for regularization
    Dense(units=1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
print("Компиляция модели...")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Обучение модели...")
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

# Evaluate the model
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
