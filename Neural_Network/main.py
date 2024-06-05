import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Activation_ReLU import Activation_ReLU
from Layer_Dense import Layer_Dense
from Activation_Softmax import Activation_Softmax

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
Activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
Activation2 = Activation_Softmax()

dense1.forward(X)
Activation1.forward(dense1.output)

dense2.forward(Activation1.output)
Activation2.forward(dense2.output)

print(Activation2.output[:5])
