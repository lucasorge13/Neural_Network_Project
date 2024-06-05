import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Activation_ReLU import Activation_ReLU
from Layer_Dense import Layer_Dense

nnfs.init()

#input data for the sample network
X = [[1, 2, 3, 2.5],
      [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)


layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)

print(layer1.output)
