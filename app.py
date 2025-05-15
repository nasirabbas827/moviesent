from flask import Flask, request, render_template
import re
import joblib
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models and tokenizer once at startup
lr_model = joblib.load('models/logistic_regression_model.pkl')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
lstm_model = load_model('models/lstm_model.h5')

with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 200

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text, tokenizer, max_len):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')
    return padded_seq

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        comment = request.form['comment']

        # Logistic Regression prediction
        cleaned_comment = clean_text(comment)
        comment_vector = tfidf_vectorizer.transform([cleaned_comment])
        lr_pred = lr_model.predict(comment_vector)[0]
        lr_sentiment = 'Positive' if lr_pred == 1 else 'Negative'

        # LSTM prediction
        lstm_input = preprocess_text(comment, tokenizer, max_len)
        lstm_prob = lstm_model.predict(lstm_input)[0][0]
        lstm_sentiment = 'Positive' if lstm_prob > 0.5 else 'Negative'

        return render_template('index.html', 
                               lr_result=lr_sentiment, 
                               lstm_result=lstm_sentiment, 
                               comment=comment)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
