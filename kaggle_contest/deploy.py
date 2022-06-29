import pandas # for loading csv files
import string # for string manipulation
from nltk.corpus import stopwords # stopwords are words that do not add much meaning to a sentence
from nltk.stem import WordNetLemmatizer # for word conversion into matching root forms
from sklearn.feature_extraction.text import CountVectorizer # transforms documents to feature vectors
from sklearn.naive_bayes import MultinomialNB # Naive Bayes classifier
from sklearn.externals import joblib
import pickle

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

trainSet = pandas.read_csv(r"C:\Users\Filip\Desktop\Praksa\spooky-author-identification\train.csv")

X = trainSet['text']
y = trainSet['author']

vectorizer = CountVectorizer(analyzer = processText).fit(X)
finalBag = vectorizer.transform(X)
finalModel = MultinomialNB().fit(finalBag, y)

joblib.dump(finalModel, 'model.pkl')

pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))