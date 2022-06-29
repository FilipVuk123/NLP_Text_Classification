import flask
import torch
import pickle
import os
import re
from model import BertClassifier

model1 = None
model2 = None
tokenizer = None

pathToModel1 = "models/EM_BO_O_bert_best.pt"
pathToModel2 = "models/KP_JB_SW_best_bert.pt"

app = flask.Flask(__name__)
port = int(os.getenv("PORT", 9090))

@app.route('/', methods=['GET'])
def hello():
   return flask.render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input = [str(s) for s in flask.request.form.values()]
    if len(input) == 2:
        out = input[0]
        tweet = input[1]
    else:
        tweet = input[0]
        out = "model1"
    token = tokenizer(re.sub(r"http\S+", '', tweet),  padding='max_length', max_length = 512, truncation=True,
                     return_tensors="pt")
    
    mask = token['attention_mask']
    token = token['input_ids']

    if (out == "model1"):
        labels1 = {'elonmusk': 0, 'Oprah': 1, 'BarackObama': 2}
        author = list(labels1.keys())[list(labels1.values()).index(model1(token, mask).argmax(dim=1).item())]
        # output = model1(token, mask).argmax(dim=1).item()
    else: 
        labels2 = {'JoeBiden': 0, 'katyperry': 1, 'stevewoz': 2}
        author = list(labels2.keys())[list(labels2.values()).index(model2(token, mask).argmax(dim=1).item())]
        # output = model2(token, mask).argmax(dim=1).item()
    
    return flask.render_template("index.html", prediction_text = "The author of tweet: {} is {}".format(tweet, author))

if __name__ == "__main__":
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    model1 = BertClassifier()
    model1.load_state_dict(torch.load(pathToModel1))
    model1.eval()
    model2 = BertClassifier()
    model2.load_state_dict(torch.load(pathToModel2))
    model2.eval()
    
    app.run(host = "127.0.0.1", port = port)

