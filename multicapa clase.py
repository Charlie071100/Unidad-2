import numpy as np

x = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
y = np.array([[0], [0], [0], [1], [1], [1]])

np.random.seed(1)  # Para reproducibilidad

w1 = np.random.rand(2, 3)  # Pesos iniciales para capa oculta
b1 = np.zeros((1, 3))  # Bias inicial para capa oculta

w2 = np.random.rand(3, 1)  # Pesos para la capa de salida
b2 = np.zeros((1, 1))  # Bias para la capa de salida


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

lr = 0.1
epoch = 500

for e in range(epoch):
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot (a1,w2) + b2
    a2 = sigmoid (z2)
    

    print ("Salida predicha: ", "\n", a2)
    print ("Salida esperada: ", "\n", y)


    error = y - a2


    d2 = error + sigmoid_derivative(a2)
    