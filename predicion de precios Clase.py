# Importamos la librería NumPy, que sirve para trabajar con listas de números (matrices) de forma muy rápida
import numpy as np

# 'x' es nuestra tabla de datos de entrada (características de las casas que ya conocemos)
# Cada fila representa una casa, y cada columna es una característica:
# - Columna 1: Tamaño de la casa en metros cuadrados (m²)
# - Columna 2: Cantidad de habitaciones
# - Columna 3: Antigüedad de la casa en años
x = np.array([
    [120,3,5],   # Casa 1: 120m², 3 habs, 5 años
    [80,2,10],   # Casa 2: 80m², 2 habs, 10 años
    [200,4,2],   # Casa 3: 200m², 4 habs, 2 años
    [150,3,7],   # ...
    [60,1,15],
    [100,2,12],
    [180,4,4],
    [90,2,8],
    [220,5,3],
    [130,3,5],
    [70,1,14],
    [160,3,5],
    [140,3,9]
],dtype=float)

# 'y' es el precio real (en miles de dólares, por ejemplo) de cada una de las casas de la tabla 'x'
# La primera casa de 120m² vale 200, la segunda de 80m² vale 120, y así sucesivamente.
y = np.array([
    [200],
    [120],
    [400],
    [250],
    [80],
    [100],
    [350],
    [150],
    [450],
    [220],
    [90],
    [200],
    [180]
], dtype=float)

# --- NORMALIZACIÓN DE DATOS ---
# Las redes neuronales se confunden si les damos números muy grandes (como 220 m²) mezclados con números pequeños (como 3 habitaciones).
# Por eso, transformamos todos los números para que estén en una escala similar (cercanos a 0).
media_x = x.mean(axis=0)  # Calculamos el promedio de cada columna
desv_x = x.std(axis=0)    # Calculamos la desviación estándar (qué tan esparcidos están los números) de cada columna
x = (x -  media_x) / desv_x  # Restamos el promedio y dividimos entre la desviación para normalizar los datos

# --- CONFIGURACIÓN DE LA RED NEURONAL ---
np.random.seed (42)  # Esto hace que los números "aleatorios" siempre sean los mismos al ejecutar, para que el resultado sea reproducible

# Los 'pesos' (w) y 'sesgos' o 'biases' (b) son los números que la red ajustará para aprender a calcular los precios.
# Piensa en ellos como perillas que la red irá girando hasta encontrar el valor correcto.

# Capa Oculta (con 10 neuronas ocultas para procesar la información):
# w1 conecta las 3 características de entrada con las 10 neuronas ocultas. Tamaño: (3 filas, 10 columnas)
w1 = np.random.randn(3,10)  
# b1 es el sesgo de la capa oculta (un valor inicial para cada una de las 10 neuronas). Tamaño: (1 fila, 10 columnas)
b1 = np.zeros((1,10))

# Capa de Salida (1 neurona final que nos dará el precio estimado):
# w2 conecta las 10 neuronas ocultas con la neurona de salida. Tamaño: (10 filas, 1 columna)
w2 = np.random.randn(10,1)
# b2 es el sesgo de la capa de salida. Tamaño: (1 fila, 1 columna)
b2 = np.zeros((1,1))

# --- FUNCIONES DE ACTIVACIÓN ---
# ReLU (Unidad Lineal Rectificada): Es una función que deja pasar los números positivos igual y convierte los negativos en 0.
# Esto ayuda a la red a entender relaciones complejas que no son líneas rectas.
def relu(x):
    return np.maximum(0,x)

# Derivada de la función ReLU: Se utiliza en el paso hacia atrás para saber cómo cambia la salida de ReLU respecto a su entrada.
# Retorna 1 si el número es mayor que 0, y 0 si es menor o igual a 0.
def relu_deriv(x):
    return (x>0).astype(float)

# lr (Learning Rate / Tasa de Aprendizaje): Es qué tan grandes serán los pasos que dará la red al ajustar sus pesos.
# Si es muy grande, la red puede pasarse del objetivo; si es muy chico, tardará una eternidad en aprender.
lr = 0.003

# --- BUCLE DE ENTRENAMIENTO (CICLO DE APRENDIZAJE) ---
# Vamos a hacer que la red practique y aprenda de sus errores 50,000 veces (épocas)
for epoch in range (50000):
    
    # 1. PROPAGACIÓN HACIA ADELANTE (Hacer la predicción con lo que sabe actualmente)
    z1 = x @ w1 + b1      # Multiplicamos las entradas normalizadas por los primeros pesos y sumamos el sesgo
    a1 = relu(z1)         # Aplicamos la función ReLU para activar las neuronas de la capa oculta
    z2 = a1 @ w2 + b2     # Multiplicamos el resultado anterior por los segundos pesos y sumamos el sesgo de salida
    y_pred = z2           # El precio que predice la red en esta iteración
    
    # 2. CÁLCULO DEL ERROR
    # Usamos el Error Cuadrático Medio: restamos el precio real (y) menos la predicción (y_pred), lo elevamos al cuadrado y sacamos el promedio.
    error = np.mean ((y-y_pred)**2)
    
    # 3. PROPAGACIÓN HACIA ATRÁS (Analizar en qué nos equivocamos y calcular gradientes)
    # Calculamos qué tanto influyó cada peso en el error final (derivadas matemáticas)
    derror = 2*(y_pred - y)/y.shape[0]  # Derivada del error respecto a la salida
    dw2 = a1.T @ derror                 # Gradiente para los pesos de la capa de salida (w2)
    db2 =  np.sum (derror, axis=0, keepdims=True)  # Gradiente para el sesgo de salida (b2)
    
    da1 = derror @ w2.T                 # Propagamos el error hacia la capa oculta
    dz1 = da1 * relu_deriv(z1)          # Deshacemos la activación ReLU en el gradiente
    dw1 = x.T @ dz1                     # Gradiente para los pesos de la primera capa (w1)
    db1 = np.sum(dz1, axis=0, keepdims=True)  # Gradiente para el sesgo de la primera capa (b1)
    
    # 4. ACTUALIZACIÓN DE LOS PESOS (Girar las perillas un poquito hacia la respuesta correcta)
    # Restamos el gradiente multiplicado por la tasa de aprendizaje (Gradient Descent)
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2
    
    # Cada 2,000 iteraciones mostramos en pantalla el error para ver si la red está aprendiendo
    if epoch % 2000 == 0:
        print(f"Época {epoch}, error actual: {error}")
        
# --- PREDICCIÓN CON EL MODELO YA ENTRENADO ---
# Ahora que la red ha aprendido, le damos los datos de una nueva casa que no estaba en la base de datos:
# Queremos saber el precio de una casa de 120m², con 3 habitaciones y 5 años de antigüedad.
nueva_casa = np.array([[120, 3, 5]], dtype=float)

# IMPORTANTE: Debemos normalizar los datos de la nueva casa usando la misma media y desviación que usamos para entrenar la red.
nueva_casa_norm = (nueva_casa - media_x) / desv_x

# Hacemos la propagación hacia adelante una única vez usando los pesos finales ajustados
z1 = nueva_casa_norm @ w1 + b1
a1 = relu(z1)
z2 = a1 @ w2 + b2
pred = z2

# Mostramos el resultado final en la pantalla
print("Precio estimado: ", pred[0][0])