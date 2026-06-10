import cv2, os, numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#\Convulciones\Reconocimiento_de_numeros <- (nombre de carpeta)
#formato onehot

RUTA_DATOS = "/home/carlosorozco/Escritorio/Sistemas inteligentes/Unidad 2/images/images"

IMAGEN_ANCHO = 64
IMAGEN_ALTO = 64

imagenes = []
etiquetas = []

for nombre_carpeta in os.listdir(RUTA_DATOS):
    ruta_carpeta = os.path.join(RUTA_DATOS, nombre_carpeta)
    if os.path.isdir(ruta_carpeta):
        for nombre_imagen in os.listdir(ruta_carpeta):
            ruta_imagen = os.path.join(ruta_carpeta, nombre_imagen)
            imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
            if imagen is not None:
                imagen = cv2.resize(imagen, (IMAGEN_ANCHO, IMAGEN_ALTO))
                imagen = imagen / 255.0
                imagenes.append(imagen)
                etiquetas.append(int(nombre_carpeta))
imagenes = np.array(imagenes).reshape(-1, IMAGEN_ANCHO, IMAGEN_ALTO, 1)
etiquetas = to_categorical(etiquetas, num_classes=10)

x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)

modelo = Sequential()
modelo.add(Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=(IMAGEN_ANCHO, IMAGEN_ALTO, 1))) #capa de convolucion -> 32 filtros, tamaño del kernel de 3x3, función de activación ReLU, y la forma de entrada de las imágenes (64x64 con 1 canal para escala de grises)
modelo.add(MaxPooling2D(pool_size=(2,2)))
modelo.add(Conv2D(64, kernel_size   =(3,3), activation='relu')) #segunda capa de convolucion -> 64 filtros, tamaño del kernel de 3x3, función de activación ReLU
modelo.add(MaxPooling2D(pool_size=(2,2)))
modelo.add(Flatten())
modelo.add(Dense(128, activation='relu'))
modelo.add(Dense(10, activation='softmax'))
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

modelo.fit(x_entrenamiento, y_entrenamiento, epochs=10, validation_data=(x_prueba, y_prueba))


modelo.save(r"/home/carlosorozco/Escritorio/Sistemas inteligentes/Unidad 2/modelo_numeros_clase.keras")
print("Modelo Guardado")