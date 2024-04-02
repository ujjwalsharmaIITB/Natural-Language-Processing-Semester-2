
from flask import Flask, request, jsonify, render_template
import spacy
import numpy as np
from flask_cors import CORS, cross_origin
nlp = spacy.load("en_core_web_sm")




app = Flask(__name__)

CORS(app, support_credentials=True)




def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def one_hot(n): #one-hot encodes the [1,2,3,4] pos tags
  z = np.zeros(4)
  if n > 0 and n < 5:
    z[n-1] = 1
  return z

def encode(sent,n):  #prepares the input_vector. Start token = 1 if starting word, else 0. If Start token 1, next 4 entries are zero since no previous word. If start token 0, next 4 tokens = previous word pos tag and then the 4 after = current word pos tag.
  if n == 0:
    start = np.array([1])
    prev = one_hot(-1)
  else:
    start = np.array([0])
    prev = one_hot(sent[n-1])

  x = np.concatenate((start,prev,one_hot(sent[n])))
  return x

def code_tag(tag):
  if tag.lower() == "noun" or tag.lower() == "propn":
    return 1
  elif tag.lower() == "det":
    return 2
  elif tag.lower() == "adj":
    return 3
  else:
    return 4

def custom_test(sent):
    W = np.array([6.00525099, -0.25138642, -2.41375883, -1.35789249,  2.7089287,  -2.79950985, 1.83492126, -2.40748616,  1.8481467,   0.23551078])
    b = np.array([-0.5396426])
    doc = nlp(sent)
    pos_tags = []
    pos_tags_orig = []
    for token in doc:
      pos_tags.append(code_tag(token.pos_))
      pos_tags_orig.append(token.pos_)
    chunk_tags = []
    for n in range(len(pos_tags)):
      c = sigmoid(np.dot(W[:9],encode(pos_tags,n)) - b)
      if c > 0.5:
        chunk_tags.append(1)
      else:
        chunk_tags.append(0)
    return chunk_tags , pos_tags_orig , pos_tags


def get_prediction(sentence):
    predictions , pos_tags_orig , pos_tags = custom_test(sentence)

    name_preds = []
    for word,pred in zip(sentence.split(),predictions):
       name_preds.append(f"{word}_{pred}")

    pos_preds = []
    for word,tag in zip(sentence.split(),pos_tags_orig):
       pos_preds.append(f"{word}_{tag}")

    code_preds = []
    for word,tag in zip(sentence.split(),pos_tags):
       code_preds.append(f"{word}_{tag}")
    
    return " ".join(name_preds) , " ".join(pos_preds) , " ".join(code_preds)




@app.get("/")
def returnHomePage():
    return render_template("index.html")


@app.get("/predict/<sentence>")
def returnPrediction(sentence):
    sentence = sentence.strip()
    prediction, pos_preds, code_preds = get_prediction(sentence)
    return jsonify({
        'prediction' : prediction,
        "pos_tags" : pos_preds,
        "code_tags" : code_preds
    })

