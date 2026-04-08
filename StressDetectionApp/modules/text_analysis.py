import joblib
import re

model = joblib.load("models/text_model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")


def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text


def analyze_text(text):

    text = clean_text(text)

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)

    if prediction[0] == 0:
        return 0
    else:
        return 2