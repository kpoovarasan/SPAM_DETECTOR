from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from joblib import load

import nltk
nltk.download('punkt', force=True)



nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
with open("Email_Spam3_RF.pkl", "rb") as file:
    model = pickle.load(file)

with open("TfidfVectorizer3.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Text preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered)

@app.route('/', methods=["GET", "POST"])
def home():
    prediction = None
    email = ""
    if request.method == "POST":
        email = request.form["email"]
        processed = preprocess_text(email)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]
    return render_template("index.html", prediction=prediction, email=email)

if __name__ == '__main__':
    app.run(debug=True)
