from flask import Flask, render_template, request

from modules.text_analysis import analyze_text
from modules.voice_analysis import detect_voice_stress
from modules.stress_classifier import classify_stress

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    text = request.form["text"]

    text_score = analyze_text(text)

    voice_score = 0
    face_score = 0

    level, solution = classify_stress(face_score, text_score, voice_score)

    return render_template("result.html",
                           level=level,
                           solution=solution)


if __name__ == "__main__":
    app.run(debug=True)