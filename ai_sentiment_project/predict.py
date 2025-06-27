import sys
import joblib

# Load model and vectorizer
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/tfidf.pkl')

# Get input review from command-line
if len(sys.argv) < 2:
    print("Please provide a review to analyze.")
    sys.exit()

review = sys.argv[1]

# Transform and predict
X = vectorizer.transform([review])
prob = model.predict_proba(X)[0]
label = model.predict(X)[0]
sentiment = "positive" if label == 1 else "negative"

print(f"Sentiment: {sentiment} (confidence: {max(prob):.2f})")
