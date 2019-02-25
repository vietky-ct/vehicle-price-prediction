import os
import pickle
import re

from flask import Flask, request, jsonify

from nltk.stem import PorterStemmer

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# Unpickle the trained classifier and write preprocessor method used
def tokenizer(text):
    return text.split(' ')

def preprocessor(text):
    # Return a cleaned version of text
    
    # Remove HTML markup
    text = re.sub('<[^>]*>', '', text)
    # Save emoticons for later appending
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove any non-word character and append the emoticons,
    # removing the nose character for standarization. Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))

    return text

# Uncomment this line after you trained your model and copied it to the same folder with app.py
#tweet_classifier = pickle.load(open('logisticRegression.pkl', 'rb'))

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return app.send_static_file('html/index.html')


@app.route('/classify', methods=['POST'])
def classify():
    text = request.form.get('text', None)
    assert text is not None

    # Take this if-statement out and apply your model here
    if 'love' in text:
        prob_neg, prob_pos = 0.1, 0.9
    elif 'hate' in text:
        prob_neg, prob_pos = 0.9, 0.1
    else:
        prob_neg, prob_pos = 0.5, 0.5

    s = 'Positive' if prob_pos >= prob_neg else 'Negative'
    p = prob_pos if prob_pos >= prob_neg else prob_neg
    return jsonify({
        'sentiment': s,
        'probability': p
    })

app.run()
