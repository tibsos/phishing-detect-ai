from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
import numpy as np
import re
from sklearn.model_selection import train_test_split
import mailbox
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load emails from both phishing and legitimate email mbox files
def load_emails(phishing_file_path, legitimate_file_path):
    print("Loading phishing emails...")
    phishing_mbox = mailbox.mbox(phishing_file_path)
    phishing_emails = []
    for message in phishing_mbox:
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    phishing_emails.append(part.get_payload(decode=True).decode())
        else:
            phishing_emails.append(message.get_payload(decode=True).decode())

    print("Loading legitimate emails...")
    legitimate_mbox = mailbox.mbox(legitimate_file_path)
    legitimate_emails = []
    for message in legitimate_mbox:
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    legitimate_emails.append(part.get_payload(decode=True).decode())
        else:
            legitimate_emails.append(message.get_payload(decode=True).decode())

    return phishing_emails, legitimate_emails

# Preprocess the emails
def preprocess_emails(emails):
    print("Preprocessing emails...")
    cleaned_emails = []
    for email in emails:
        email = re.sub(r'\W', ' ', email)  # Remove special characters
        email = email.lower()  # Convert to lowercase
        cleaned_emails.append(email)
    return cleaned_emails

# Load and preprocess the emails
print("Loading and preprocessing emails...")
phishing_emails, legitimate_emails = load_emails('emails-phishing.mbox', 'emails-normal.mbox')
emails = phishing_emails + legitimate_emails
cleaned_emails = preprocess_emails(emails)

# Create labels for the dataset (1 for phishing, 0 for non-phishing)
labels = np.array([1] * len(phishing_emails) + [0] * len(legitimate_emails))

# Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(cleaned_emails, labels, test_size=0.2, random_state=42)

# Tokenization and padding
print("Tokenizing and padding sequences...")
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform input size
X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

# Define the neural network model
print("Defining the neural network model...")
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
print("Compiling the model...")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test))

model.save_weights('/mnt/data/model_weights.h5')

# Testing the model on some emails
def test_model(model, phishing_emails, legitimate_emails):
    print("Testing the model...")
    # Load a few phishing and legitimate emails for testing
    test_phishing = phishing_emails[:100]
    test_legitimate = legitimate_emails[:100]

    # Combine and preprocess test emails
    test_emails = test_phishing + test_legitimate
    cleaned_test_emails = preprocess_emails(test_emails)

    # Tokenize and pad test emails
    test_seq = tokenizer.texts_to_sequences(cleaned_test_emails)
    X_test_pad = pad_sequences(test_seq, maxlen=100)

    # Make predictions
    predictions = model.predict(X_test_pad)

    # Evaluate the predictions
    for i, prediction in enumerate(predictions):
        if prediction >= 0.5:
            print(f"Email {i+1} is predicted as phishing with probability: {prediction[0]:.2f}")
        else:
            print(f"Email {i+1} is predicted as legitimate with probability: {1-prediction[0]:.2f}")

# Test the model
test_model(model, phishing_emails, legitimate_emails)

print("Process completed.")
