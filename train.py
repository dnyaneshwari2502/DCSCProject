# train.py

import pandas as pd
import numpy as np
import re
import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

print("--- Starting Model Training Script ---")

# --- 1. Configuration (Consolidated from your notebook) ---
# Use relative paths for portability
BASE_DATA_PATH = 'Data' 
TRAIN_FOLDER1_NAME = 'clickbait17-train-170331'
INSTANCES_FILE_NAME = 'instances.jsonl'
TRUTH_FILE_NAME = 'truth.jsonl'
GLOVE_FILE_PATH = os.path.join(BASE_DATA_PATH, 'glove.twitter.27B', 'glove.twitter.27B.100d.txt')

# Paths for saving artifacts
MODEL_ARTIFACTS_PATH = 'app/models'
os.makedirs(MODEL_ARTIFACTS_PATH, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_ARTIFACTS_PATH, 'multi_output_model.h5')
TOKENIZER_SAVE_PATH = os.path.join(MODEL_ARTIFACTS_PATH, 'tokenizer.pkl')
CLICKBAIT_ENCODER_SAVE_PATH = os.path.join(MODEL_ARTIFACTS_PATH, 'clickbait_encoder.pkl')
EMOTION_ENCODER_SAVE_PATH = os.path.join(MODEL_ARTIFACTS_PATH, 'emotion_encoder.pkl')


# Model & Preprocessing Parameters
VALIDATION_SET_SIZE = 0.20
RANDOM_STATE = 42
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
LSTM_UNITS = 128
DROPOUT_RATE = 0.40
EMBEDDING_TRAINABLE = True
EPOCHS = 15
BATCH_SIZE = 64
PATIENCE = 3
LEARNING_RATE = 0.001

# --- 2. NLTK Downloads & Preprocessing Functions ---
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

def safe_get_text(text_list_input):
    if isinstance(text_list_input, list) and len(text_list_input) > 0 and isinstance(text_list_input[0], str):
        return text_list_input[0]
    elif isinstance(text_list_input, str):
        return text_list_input
    return ""

def preprocess_text(text_input):
    text = safe_get_text(text_input).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[@#]\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOP_WORDS and len(word) > 1]
    return cleaned_tokens

# --- 3. Data Loading (Adapted from your notebook) ---
print("Loading and merging data...")
def load_and_merge_data(folder_path):
    instances_path = os.path.join(folder_path, INSTANCES_FILE_NAME)
    truth_path = os.path.join(folder_path, TRUTH_FILE_NAME)
    if not os.path.exists(instances_path) or not os.path.exists(truth_path):
        return pd.DataFrame()
    instances_df = pd.read_json(instances_path, lines=True, encoding='utf-8')
    truth_df = pd.read_json(truth_path, lines=True, encoding='utf-8')
    return pd.merge(instances_df, truth_df, on='id', how='inner')

train_path1 = os.path.join(BASE_DATA_PATH, TRAIN_FOLDER1_NAME)
df = load_and_merge_data(train_path1)

if df.empty:
    raise ValueError("Data loading failed. DataFrame is empty.")

print(f"Loaded {len(df)} records.")

# --- 4. Feature Engineering and Preprocessing ---
print("Preprocessing text data...")
df['emotion_label'] = df['truthClass'].apply(lambda x: 'sensational_emotion' if x == 'clickbait' else 'neutral_emotion')
df['cleaned_tokens'] = df['postText'].apply(preprocess_text)

# --- 5. Label Encoding and Saving Encoders ---
print("Encoding labels...")
clickbait_encoder = LabelEncoder()
df['clickbait_numeric'] = clickbait_encoder.fit_transform(df['truthClass'])

emotion_encoder = LabelEncoder()
df['emotion_numeric'] = emotion_encoder.fit_transform(df['emotion_label'])

print("Saving label encoders...")
with open(CLICKBAIT_ENCODER_SAVE_PATH, 'wb') as f:
    pickle.dump(clickbait_encoder, f)
with open(EMOTION_ENCODER_SAVE_PATH, 'wb') as f:
    pickle.dump(emotion_encoder, f)

# --- 6. Train/Validation Split ---
print("Splitting data into training and validation sets...")
X = df['cleaned_tokens']
y_df = df[['clickbait_numeric', 'emotion_numeric']]

X_train_tokens, X_valid_tokens, y_train_df, y_valid_df = train_test_split(
    X, y_df, test_size=VALIDATION_SET_SIZE, random_state=RANDOM_STATE, stratify=df['clickbait_numeric']
)

# --- 7. Undersampling (Balancing the training set) ---
print("Balancing the training set via undersampling...")
train_data_full = pd.DataFrame({
    'cleaned_tokens': X_train_tokens,
    'clickbait_numeric': y_train_df['clickbait_numeric'],
    'emotion_numeric': y_train_df['emotion_numeric']
})
df_minority = train_data_full[train_data_full['clickbait_numeric'] == 0]
df_majority = train_data_full[train_data_full['clickbait_numeric'] == 1]
df_majority_undersampled = df_majority.sample(n=len(df_minority), random_state=RANDOM_STATE)
train_df_balanced = pd.concat([df_minority, df_majority_undersampled]).sample(frac=1, random_state=RANDOM_STATE)

X_train_tokens_balanced = train_df_balanced['cleaned_tokens']
y_train_clickbait_balanced = train_df_balanced['clickbait_numeric']
y_train_emotion_balanced = train_df_balanced['emotion_numeric']

print(f"Balanced training set size: {len(train_df_balanced)}")

# --- 8. Tokenization and Padding & Saving Tokenizer ---
print("Fitting tokenizer and padding sequences...")
X_train_texts = X_train_tokens.apply(lambda t: ' '.join(t)) # Fit on original for broader vocab
X_train_texts_balanced = X_train_tokens_balanced.apply(lambda t: ' '.join(t))
X_valid_texts = X_valid_tokens.apply(lambda t: ' '.join(t))

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_texts)

print("Saving tokenizer...")
with open(TOKENIZER_SAVE_PATH, 'wb') as f:
    pickle.dump(tokenizer, f)

X_train_padded_balanced = pad_sequences(tokenizer.texts_to_sequences(X_train_texts_balanced), maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_valid_padded = pad_sequences(tokenizer.texts_to_sequences(X_valid_texts), maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

y_train_labels = [y_train_clickbait_balanced.to_numpy(), y_train_emotion_balanced.to_numpy()]
y_valid_labels = [y_valid_df['clickbait_numeric'].to_numpy(), y_valid_df['emotion_numeric'].to_numpy()]


# --- 9. GloVe Embeddings ---
print("Loading GloVe embeddings...")
embeddings_index = {}
with open(GLOVE_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

word_index = tokenizer.word_index
effective_vocab_size = min(VOCAB_SIZE, len(word_index) + 1)
embedding_matrix = np.zeros((effective_vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < effective_vocab_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# --- 10. Build and Train the Multi-Output Model ---
print("Building the multi-output LSTM model...")
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = Embedding(effective_vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=EMBEDDING_TRAINABLE)(sequence_input)
lstm_out = Bidirectional(LSTM(LSTM_UNITS, return_sequences=False))(embedding_layer)
dropout_layer = Dropout(DROPOUT_RATE)(lstm_out)

clickbait_output = Dense(1, activation='sigmoid', name='clickbait_output')(dropout_layer)
emotion_output = Dense(1, activation='sigmoid', name='emotion_output')(dropout_layer)

model = Model(inputs=sequence_input, outputs=[clickbait_output, emotion_output])
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss={'clickbait_output': 'binary_crossentropy', 'emotion_output': 'binary_crossentropy'},
    metrics={'clickbait_output': 'accuracy', 'emotion_output': 'accuracy'}
)
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

print("\nTraining the model...")
model.fit(
    X_train_padded_balanced,
    {'clickbait_output': y_train_labels[0], 'emotion_output': y_train_labels[1]},
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_valid_padded, {'clickbait_output': y_valid_labels[0], 'emotion_output': y_valid_labels[1]}),
    callbacks=[early_stopping],
    verbose=2
)

# --- 11. Save the Trained Model ---
print(f"Saving trained model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)

print("\n--- Training script finished successfully! ---")