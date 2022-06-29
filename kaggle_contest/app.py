import flask
import os
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.externals import joblib

app = flask.Flask(__name__)
port = int(os.getenv("PORT", 9090))

def processText(text):
    
    # punctuation removal
    text = ''.join(c for c in text if c not in string.punctuation)
    
    # lemmatisation on verbs, nouns and adjectives
    lemmatiser = WordNetLemmatizer() # converts words into root forms
    
    final, step1, step2, lemmahelp = '', '', '', ''
    
    for i in range(len(text.split())): # split by ' '
        lemmahelp = lemmatiser.lemmatize(text.split()[i], pos = 'v') 
        step1 += lemmahelp + ' '
   
    for i in range(len(step1.split())): # split by ' '
        lemmahelp = lemmatiser.lemmatize(step1.split()[i], pos = 'n') 
        step2 += lemmahelp + ' '
    
    for i in range(len(step2.split())): # split by ' '
        lemmahelp = lemmatiser.lemmatize(step2.split()[i], pos = 'a') 
        final += lemmahelp + ' '
        
    # stopwords removal
    return [w for w in final.split() if w.lower() not in stopwords.words('english')]

model = joblib.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/', methods=['GET'])
def hello():
   return "Hello world!"

@app.route('/predict', methods=['POST'])
def predict():
    quote = flask.request.get_json(force=True)['quote']
    print("quote = ", quote)
    quote = vectorizer.transform([quote])

    prediction = model.predict(quote)
    print(prediction)
    return flask.jsonify(prediction[0])

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=port)