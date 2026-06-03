import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

w = np.random.rand(2)
b = np.random.rand(1)
lr = 0.2  # tasa de aprendizaje

print("Pesos iniciales:", w, "Bias:", b)

def predecir(x_in):
    z = np.dot(x_in, w) + b
    if z > 0.5:
        return 1
    return 0

for epoch in range(50):
    for i in range(len(x)):
        z = np.dot(x[i], w) + b
        if z > 0.5:
            y_pred = 1
        else:
            y_pred = 0

        error = y[i] - y_pred
        w += lr * error * x[i]
        b += lr * error

    print(f"Epoch {epoch + 1} - Pesos: {w} Bias: {b}")

print("Resultados finales:")
for i in range(len(x)):
    print(f"Entrada: {x[i]} ||| Salida esperada: {y[i]} ||| Salida predicha: {predecir(x[i])}")

nuevo = np.array([1, 1])
print(f"Ingresaste {nuevo} y su resultado es {predecir(nuevo)}")