
import numpy as np
from numpy import exp, log2
import csv
import tqdm
from tqdm.notebook import tqdm_notebook
import random
import pickle

import __main__

np.random.seed(42)



# Linear Class


class Linearlayer():
    def __init__(self,input_neurons,output_neurons , name = "layer", momentum = None):
        self.weights = np.random.randn(input_neurons,output_neurons)
        self.bias = np.zeros((1,output_neurons))
        self.name = name
        self.momentum = momentum
        self.delta_weights = 0
        self.delta_bias = 0


    def forward(self,inputs):
        self.input = inputs
        # print(f"shape at {self.name}" , inputs.shape)
      # Input is in the form 1*n , weights is n*o so dot product is 1*o
        self.output = np.dot(inputs,self.weights) + self.bias
        return self.output

    # given dE/dnet_layer = delta it calculates dE/dW
    def backward(self, deltas, lr):
      # print(f"starting backward for {self.name}")
      # print(deltas.shape)
      # deltas is of the shape = 1*out, inputs of the shape in*1
      d_error_by_d_weights = np.dot(self.input.T , deltas)
      # in*out matrix

      # d_error_by_d_input will become dE/dnet_layer for previous layer 1*in
      d_error_by_d_input = np.dot(deltas , self.weights.T)
      # 1*out , out*in = 1*in

      d_error_by_d_bias = 1*deltas
      # 1*out

      # Weight Changes
      # check if momentum is there
      if self.momentum is None:
        self.weights -= lr * d_error_by_d_weights
        self.bias -= lr * d_error_by_d_bias
      else:
        self.delta_weights =  lr * d_error_by_d_weights + self.momentum * self.delta_weights
        self.weights -= self.delta_weights

        self.delta_bias = lr * d_error_by_d_bias + self.momentum * self.delta_bias
        self.bias -= self.delta_bias

      # print(f"done backward for {self.name}")

      return d_error_by_d_input




__main__.Linearlayer = Linearlayer


# Activation Function
    

class ActivationLayer():
  def __init__(self, name='layer' , activation="sigmoid"):
    self.name = name
    self.activation = activation

  def forward(self, x):
    self.x = x
    if self.activation == "sigmoid":
      return self._sigmoid_activation(x)

    if self.activation == "relu":
      return self._RELU_activation(x)

    return self._softmax_activation(x)

  def _sigmoid_activation(self , x):
    return 1/(1+exp(-x))

  def _derivative_sigmoid_activation(self , x):
    x = self._sigmoid_activation(x)
    return x*(1-x)

  def _softmax_activation(self , x):
    x = exp(x) / np.sum(exp(x))


  def _derivative_softmax_activation(self, x):
    S = self._softmax_activation(x)
    S_vector = S.reshape(S.shape[0],1)
    S_matrix = np.tile(S_vector,S.shape[0])
    S_dir = np.diag(S) - (S_matrix * np.transpose(S_matrix))
    return S_dir


  def _RELU_activation(self, x):
    return x * (x > 0)

  def _derivative_RELU_activation(self, x):
    return 1. * (x > 0)


  def backward(self, deltas, lr):
    # here this will pass the dE/Dnet_out_prev but act has been applied
    # so chain rule dE/net_prev = dE/dnet_out * dnet_out/dnet_prev
    # dE/dnet_prev = deltas * der_of_activation
    # * is pairwise multiplication
    # print(f"starting backward for {self.name}")
    if self.activation == "sigmoid":
      return self._derivative_sigmoid_activation(self.x) * deltas

    if self.activation == "relu":
      return self._derivative_RELU_activation(self.x) * deltas

    return self._derivative_softmax_activation(self.x) * deltas



__main__.ActivationLayer = ActivationLayer



# Loss Function
  


def binary_cross_entropy_loss_function(predicted,actual):
  if actual == 1:
    return -1 * log2(predicted)
  else:
    return -1*log2(1-predicted)
  # loss_val = -1 * actual * log2(predicted) - 1*(1-actual)*log2(1-predicted)
  # return loss_val


__main__.binary_cross_entropy_loss_function = binary_cross_entropy_loss_function


def derivative_binary_cross_entropy_loss_function(predicted , actual):
  if actual == 1:
    return -1 / predicted
  else:
    return 1/(1-predicted)


__main__.derivative_binary_cross_entropy_loss_function = derivative_binary_cross_entropy_loss_function


def weighted_binary_cross_entropy_loss_function(predicted , actual , weight_1 = 10, weight_0 = 1):
  if actual == 1:
    return -1 * weight_1 * log2(predicted)
  else:
    return -1 * weight_0 *log2(1-predicted)


__main__.weighted_binary_cross_entropy_loss_function = weighted_binary_cross_entropy_loss_function

def derivative_weighted_binary_cross_entropy_loss_function(predicted , actual , weight_1 = 5, weight_0 = 1):
  if actual == 1:
    return - weight_1 / predicted
  else:
    return weight_0 /(1-predicted)



__main__.derivative_weighted_binary_cross_entropy_loss_function = derivative_weighted_binary_cross_entropy_loss_function


def mse_error(predicted , actual):
  return np.square(predicted - actual) / 2

__main__.mse_error = mse_error

def derivative_mse_error(predicted,output):
  return predicted - output

__main__.derivative_mse_error = derivative_mse_error



# Model Class

class NeuralNet():
    def __init__(
        self, input_neurons, hidden_neuron, output_neuron, learning_rate,
        momentum = None,
        activation="sigmoid",
        loss_function = binary_cross_entropy_loss_function,
        derivative_loss_function = derivative_binary_cross_entropy_loss_function):
        self.inputs_neurons = input_neurons
        self.hidden_neuron = hidden_neuron
        self.output_neuron = output_neuron
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.activation = activation
        self.derivative_loss_function = derivative_loss_function
        self.momentum = momentum
        self.input_layer = Linearlayer(input_neurons, hidden_neuron , name="input_layer", momentum = momentum)
        self.activation1 = ActivationLayer(name="activation_layer_1" , activation = activation)
        self.output_layer = Linearlayer(hidden_neuron, output_neuron,name="output_layer", momentum = momentum)
        self.activation2 = ActivationLayer(name = "activation_layer_2" , activation="sigmoid")


    def forward(self,input_value):
        self.input_value = input_value
        # dimension is 1*O
        self.hidden_output = self.input_layer.forward(input_value)
        self.hidden_output = self.activation1.forward(self.hidden_output)
        # print(self.hidden_output)
        # dimension is 1*H

        # dimension now is 1*H
        self.output = self.output_layer.forward(self.hidden_output)
        self.output = self.activation2.forward(self.output)
        # print(self.output)
        # dimension is 1*O
        return self.output


    def predict(self , input_value):
        output = self.forward(input_value)
        output = output >= 0.5
        return output


    def train(self, inputs , outputs , epochs):
      learning_rate = self.learning_rate
      print(f"Learning rate is {learning_rate}")
      print_every = epochs // 10

      for i in tqdm_notebook( range(epochs) , desc = "Training Progress"):
        # print(f"Starting {i+1} epoch.")
        error_per_epoch = 0

        for input,output in zip(inputs,outputs):
          input = input.reshape(1 ,input.shape[0])
          out = self.forward(input)
          # calculating loss
          loss = self.loss_function(out , output)
          error_per_epoch += loss

          delta = self.derivative_loss_function(out , output)

          # doing manually but can be done via for loop
          delta = self.activation2.backward(delta , learning_rate)
          delta = self.output_layer.backward(delta , learning_rate)

          delta = self.activation1.backward(delta , learning_rate)
          delta = self.input_layer.backward(delta , learning_rate)

        # print(f"error on epoch: {i+1} is : {error_per_epoch}")
        if (i) % print_every == 0:
          print(f"error on epoch: {i} is : {error_per_epoch}")




__main__.NeuralNet = NeuralNet





def predict_output(model , example):
    example = np.array(example)
    pred = model.predict(example)
    if pred == True:
      return "Palindrome"
    return "Not Palindrome"



models_5_list = pickle.load(open("models/models_5_list.pkl" , "rb"))

model_4_list = pickle.load(open("models/models_4_list.pkl" , "rb"))

model_3_list = pickle.load(open("models/models_3_list.pkl" , "rb"))

model_2_list = pickle.load(open("models/models_2_list.pkl" , "rb"))



def predict_output_5(example):
  model = models_5_list[0]
  return predict_output(model , example)


def predict_output_4(example):
  model = model_4_list[0]
  return predict_output(model , example)

def predict_output_3(example):
  model = model_3_list[0]
  return predict_output(model , example)


def predict_output_2(example):
  model = model_2_list[0]
  return predict_output(model , example)


