from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        msg = request.form["message"]
        X = vectorizer.transform([msg])
        pred = model.predict(X)[0]
        result = "SPAM" if pred == 1 else "NOT SPAM"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

