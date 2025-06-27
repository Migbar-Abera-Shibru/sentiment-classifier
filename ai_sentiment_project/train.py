import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump
from datasets import load_dataset

# Load IMDb dataset from HuggingFace
dataset = load_dataset("imdb", split="train")
df = pd.DataFrame(dataset)

# For simplicity, reduce dataset size
df = df.sample(n=5000, random_state=42)

# Preprocessing
df['label'] = df['label'].map({1: 1, 0: 0})
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
dump(model, 'model/model.pkl')
dump(vectorizer, 'model/tfidf.pkl')
