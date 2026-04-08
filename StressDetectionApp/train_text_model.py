import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# make sure models folder exists
os.makedirs("models", exist_ok=True)

# sample dataset
data = {
    "text": [
        "I feel very stressed",
        "I am under pressure",
        "I am depressed and tired",
        "I feel relaxed and happy",
        "Today is a beautiful day",
        "I am calm and peaceful",
        "My work is overwhelming",
        "I feel anxious",
        "I am enjoying my day",
        "Everything is going well"
    ],

    "label": [1,1,1,0,0,0,1,1,0,0]
}

df = pd.DataFrame(data)

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer()

X_vector = vectorizer.fit_transform(X)

model = LogisticRegression()

model.fit(X_vector, y)

joblib.dump(model, "models/text_model.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")

print("Text model trained successfully")