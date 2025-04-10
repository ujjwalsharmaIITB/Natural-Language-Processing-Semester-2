# -*- coding: utf-8 -*-
"""CS772 - BPTT Code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uwr0e-XBE2BBOm640VBJ0Dskw8TCdW31

# Dependencies
"""

#Installing dependencies - Run if needed

# !pip install numpy==1.25.2
# !pip install tqdm==4.66.2
# !pip install scikit-learn==1.2.2
# !pip install spacy==3.7.4

"""# Data"""

import json
import numpy as np

def one_hot(n): #one-hot encodes the [1,2,3,4] pos tags
  z = np.zeros(4)
  if n > 0 and n < 5:
    z[n-1] = 1
  return z

def input(sent,n):  #prepares the input_vector. Start token = 1 if starting word, else 0. If Start token 1, next 4 entries are zero since no previous word. If start token 0, next 4 tokens = previous word pos tag and then the 4 after = current word pos tag.
  if n == 0:
    start = np.array([1])
    prev = one_hot(-1)
  else:
    start = np.array([0])
    prev = one_hot(sent[n-1])

  x = np.concatenate((start,prev,one_hot(sent[n])))
  return x

def dataloader(file): #Just write the filename in place of file and it will load the data (you have to upload them to colab first) There are two files -train and test
  data = []
  with open(file) as json_file:
      for row in json_file:
        data.append(json.loads(row))
  X = []
  y = []
  for k in range(len(data)):
    for i in range(len(data[k]['tokens'])):
      X.append(input(data[k]['pos_tags'],i))
      y.append(np.array([data[k]['chunk_tags'][i]]))
  X = np.array(X)
  y = np.array(y)

  return X,y

# Load train data
X_train, y_train = dataloader("train.jsonl")

# Load test data
X_test, y_test = dataloader("test.jsonl")

y_train

"""# Recurrent Perceptron

## Recurrent Perceptron class
"""

from tqdm import tqdm

import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

#Instead of perceptron, I'm using a sigmoid (Maybe we can make the curve steeper)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return sigmoid(x)*(1-sigmoid(x))

def code_tag(tag):
  if tag.lower() == "noun":
    return 1
  elif tag.lower() == "det":
    return 2
  elif tag.lower() == "adj":
    return 3
  else:
    return 4


class RecurrentPerceptron:
  def __init__(self, input_dim):  #input_dim = 9 and T is the number of steps after which weights get updated

    #Model Parameters

    self.W = np.random.randn(input_dim + 1) # Combine input and recurrent weights
    self.hidden_state = np.zeros(1)  # Initialize hidden state
    self.b = np.random.uniform(-1,1,1) #Initialise bias
    self.T = 3 #number of iterations before BPTT


  def forward_backward_pass(self,x,y,learning_rate):
    x = x.reshape((9))
    y = y.reshape((1))


    phi = []            #To store the hidden_state values as we forward pass
    phi_prime = []      #To store the derivate values as we forward pass

    for iterations in range(self.T):

      combined = np.concatenate((x, self.hidden_state))
      self.hidden_state = np.array(sigmoid(self.W @ combined - self.b))  # Calculate hidden state with feedback , @ is np.dot

      #Store hidden state and derivate values

      phi.append(self.hidden_state)
      phi_prime.append(np.array([sigmoid_derivative(self.W @ combined - self.b)]))


    approximation_length = 2  #Recurring gradients explode; approximation length = Till how long back should we consider


    #Complicated product and sum expression to calculate gradient. Pls try to modify/verify this.

    s = 0
    sh = 0
    check = 0
    for l in range(approximation_length):
      check = 1
      p = 1
      for k in range(1,l+2):
        p *= phi_prime[-k] # approximation - product term for last k gradients
      p *= self.W[-1]**l
      ph = p*phi[-(l+1)]
      s += p
      sh += ph
    if check != 1:
      s = 1
      sh = 1

    gradient_x = s*x*(self.hidden_state - y)                #gradient corresponding to input_weights
    gradient_h = np.array(sh*(self.hidden_state - y))       #gradient corresponding to hidden_weight
    gradient = np.concatenate((gradient_x, gradient_h), axis = 1) #concatenating to make one set of weights
    gradient_b = -s*(self.hidden_state - y)                  #gradient for bias term

    gradient = gradient.reshape(10)
    gradient_b = gradient_b.reshape(1)

    #print(gradient.shape, self.W.shape)

    #Calculate the Gradient
    self.W += -learning_rate*gradient       #Weight update
    self.b += -learning_rate*gradient_b     #Bias update




  def custom_test(self,sent):
    doc = nlp(sent)
    pos_tags = []
    for token in doc:
      pos_tags.append(code_tag(token.pos_))
    chunk_tags = []
    for n in range(len(pos_tags)):
      c = sigmoid(np.dot(self.W[:9],input(pos_tags,n)) - self.b)
      if c > 0.5:
        chunk_tags.append(1)
      else:
        chunk_tags.append(0)
    return chunk_tags
  #Function to evaluate model on a test case

  def evaluate(self,x,y):
    c = sigmoid(np.dot(self.W[:9],x) - self.b)
    y = y[0]
    if c>0.5:
      yp = 1
    else:
      yp = 0
    return int(yp == y)
  def predict(self,x):
    c = sigmoid(np.dot(self.W[:9],x) - self.b)
    if c>0.5:
      yp = 1
    else:
      yp = 0
    return int(yp)

"""## Initialising Model"""

model = RecurrentPerceptron(9)

"""### Randomize the Training set"""

indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

X_train[0]

"""##5-Fold Cross-validation"""

from sklearn.model_selection import KFold
from tqdm import tqdm

X_combined = X_train
y_combined = y_train

num_folds = 5

kf = KFold(n_splits=num_folds, shuffle=True)

fold_accuracies = []
fold_weights = []
fold_biases = []
for train_index, test_index in tqdm(kf.split(X_combined)):
    X_train_fold, X_test_fold = X_combined[train_index], X_combined[test_index]
    y_train_fold, y_test_fold = y_combined[train_index], y_combined[test_index]

    model = RecurrentPerceptron(9)
    lr = 0.2
    for index in range(len(X_train_fold)):
        x = X_train_fold[index]
        y = y_train_fold[index]
        for iterations in range(2):
            model.forward_backward_pass(x, y, lr)
        s = 0
    count = 0
    for k in range(len(X_test_fold)):
        s += (model.evaluate(X_test_fold[k], y_test_fold[k]))
        count += 1
    acc = s / count
    print(acc)
    fold_accuracies.append(acc)
    fold_weights.append(model.W)
    fold_biases.append(model.b)

k=0
for i in fold_accuracies:
  k=k+i
k=k/5
print("Average Validation Accuracy:",k)

"""## Testing"""

sum1=0
for i in range(len(X_test)):
  sum1+=model.evaluate(X_test[i],y_test[i])
sum1=sum1/(len(X_test))
print("Average Test Accuracy: ",sum1)

print("Model Weights: ", model.W)
print("Model Bias: ", model.b)

model.custom_test("The quick brown fox jumped over the lazy dog.")