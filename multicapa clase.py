import numpy as np

# Datos de entrada: cada fila es un ejemplo con dos características.
x = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])

# Salidas esperadas para cada ejemplo. Es un problema de clasificación simple.
y = np.array([[0], [0], [0], [1], [1], [1]])

np.random.seed(1)  # Fija la semilla para obtener resultados reproducibles

# Pesos y bias de la capa oculta.
# w1 tiene forma (2, 3) porque hay 2 entradas y 3 neuronas en la capa oculta.
w1 = np.random.rand(2, 3)
b1 = np.zeros((1, 3))

# Pesos y bias de la capa de salida.
# w2 tiene forma (3, 1) porque la capa oculta tiene 3 salidas y la salida final es una sola neurona.
w2 = np.random.rand(3, 1)
b2 = np.zeros((1, 1))


def sigmoid(x):
    # Función de activación sigmoide.
    # Convierte valores reales en el rango (0, 1).
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    # Derivada de la sigmoide, necesaria para ajustar los pesos.
    sx = sigmoid(x)
    return sx * (1 - sx)


lr = 0.1  # Tasa de aprendizaje
epoch = 500  # Cantidad de ciclos de entrenamiento

for e in range(epoch):
    # Propagación hacia adelante - capa oculta
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)

    # Propagación hacia adelante - capa de salida
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    # Mostrar la salida que predice la red y la salida esperada.
    print("Salida predicha:", "\n", a2)
    print("Salida esperada:", "\n", y)

    # Error simple entre el valor esperado y el valor predicho.
    error = y - a2

    # d2 representa la corrección en la capa de salida.
    # Se combina el error con la derivada de la activación para saber cuánto ajustar.
    d2 = error + sigmoid_derivative(a2)

    # d1 representa la corrección en la capa oculta.
    # Se propaga el error hacia atrás usando los pesos de la capa de salida.
    d1 = np.dot(d2, w2.T) * sigmoid_derivative(a1)

    # Actualizar pesos y bias de la capa de salida.
    # Se usa la salida de la capa oculta (a1) para calcular el gradiente.
    w2 += lr * np.dot(a1.T, d2)
    b2 += lr * np.sum(d2, axis=0, keepdims=True)

    # Actualizar pesos y bias de la capa oculta.
    # Se usa la entrada original x y el ajuste d1 para la actualización.
    w1 += lr * np.dot(x.T, d1)
    b1 += lr * np.sum(d1, axis=0, keepdims=True)