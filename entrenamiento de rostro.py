import cv2  # biblioteca OpenCV para leer y procesar imágenes
import os  # biblioteca para interactuar con el sistema de archivos
import numpy as np  # biblioteca NumPy para arreglos y cálculos numéricos
from tensorflow.keras.models import Sequential  # modelo secuencial de Keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # capas de la red neuronal

from tensorflow.keras.utils import to_categorical  # utilidad para convertir etiquetas a one-hot
from tensorflow.keras.regularizers import l2  # regularización L2 para capas densas
from sklearn.model_selection import train_test_split  # función para dividir datos en entrenamiento y prueba

from sklearn.preprocessing import LabelEncoder  # codificador de etiquetas de texto a valores numéricos
from tensorflow.keras.callbacks import EarlyStopping  # callback para detener entrenamiento temprano

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # generador para aumento de datos de imágenes


RUTA_DATOS = r"/home/carlosorozco/Escritorio/Sistemas inteligentes/Unidad 2/Reconocimiento de rostros"  # ruta donde están las carpetas de rostros

ANCHO = 256  # ancho objetivo de las imágenes
ALTO = 256  # alto objetivo de las imágenes

imagenes = []  # lista para guardar las imágenes procesadas
etiquetas = []  # lista para guardar las etiquetas correspondientes de cada imagen
nombres = []  # lista para guardar los nombres únicos de las personas

# usar imágenes sueltas si no hay carpetas por persona, o usar subcarpetas si existen
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
items = os.listdir(RUTA_DATOS)
subcarpetas = [item for item in items if os.path.isdir(os.path.join(RUTA_DATOS, item))]

if subcarpetas:
    for nombre_persona in subcarpetas:  # recorrer cada carpeta de persona
        ruta_persona = os.path.join(RUTA_DATOS, nombre_persona)
        nombres.append(nombre_persona)  # guardar el nombre de la persona
        for archivo_imagen in os.listdir(ruta_persona):  # recorrer cada archivo de imagen en esa carpeta
            ruta_imagen = os.path.join(ruta_persona, archivo_imagen)  # ruta completa de la imagen
            extension = os.path.splitext(archivo_imagen)[1].lower()
            if extension not in valid_extensions:
                continue  # ignorar archivos que no son imágenes
            imagen = cv2.imread(ruta_imagen)  # leer la imagen usando OpenCV
            if imagen is not None:  # asegurarse de que la lectura haya sido exitosa
                imagen = cv2.resize(imagen, (ANCHO, ALTO))  # redimensionar la imagen al tamaño deseado
                imagen = imagen / 255.0  # normalizar los valores de píxeles a [0, 1]
                imagenes.append(imagen)  # agregar la imagen procesada a la lista
                etiquetas.append(nombre_persona)  # agregar la etiqueta (nombre de persona) para esa imagen
else:
    for archivo_imagen in items:  # recorrer imágenes sueltas en la carpeta principal
        extension = os.path.splitext(archivo_imagen)[1].lower()
        if extension not in valid_extensions:
            continue  # ignorar archivos no válidos
        nombre_persona = os.path.splitext(archivo_imagen)[0]  # usar el nombre de archivo como etiqueta
        ruta_imagen = os.path.join(RUTA_DATOS, archivo_imagen)
        imagen = cv2.imread(ruta_imagen)  # leer la imagen usando OpenCV
        if imagen is not None:  # asegurarse de que la lectura haya sido exitosa
            imagen = cv2.resize(imagen, (ANCHO, ALTO))  # redimensionar la imagen al tamaño deseado
            imagen = imagen / 255.0  # normalizar los valores de píxeles a [0, 1]
            imagenes.append(imagen)  # agregar la imagen procesada a la lista
            etiquetas.append(nombre_persona)  # agregar la etiqueta basada en el nombre del archivo
            if nombre_persona not in nombres:
                nombres.append(nombre_persona)
imagenes = np.array(imagenes)  # convertir la lista de imágenes a un arreglo NumPy
etiquetas = np.array(etiquetas)  # convertir la lista de etiquetas a un arreglo NumPy

label_encoder = LabelEncoder()  # crear el codificador de etiquetas
etiquetas_numericas = label_encoder.fit_transform(etiquetas)  # transformar etiquetas de texto a números
etiquetas_categoricas = to_categorical(etiquetas_numericas)  # convertir etiquetas numéricas a formato one-hot

x_train, x_test, y_train, y_test = train_test_split(
    imagenes, etiquetas_categoricas, test_size=0.3, random_state=42
)  # separar datos en entrenamiento y prueba

modelo = Sequential()  # crear un modelo secuencial vacío
modelo.add(
    Conv2D(
        64,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(ALTO, ANCHO, 3),
    )
)  # primera capa convolucional con 64 filtros; usa ReLU para activar sólo valores positivos y añadir no linealidad
modelo.add(MaxPooling2D(pool_size=(2, 2)))  # capa de pooling para reducir la resolución espacial
modelo.add(
    Conv2D(
        128,
        kernel_size=(3, 3),
        activation='relu',
    )
)  # segunda capa convolucional con 128 filtros; ReLU ayuda a que la red aprenda patrones no lineales
modelo.add(MaxPooling2D(pool_size=(2, 2)))  # segunda capa de pooling
modelo.add(Flatten())  # aplanar la salida convolucional a un vector 1D
modelo.add(Dense(256, activation='relu', kernel_regularizer=l2(0.005)))  # capa densa con regularización L2 y ReLU para aprendizaje no lineal
modelo.add(Dropout(0.5))  # apagar el 50% de neuronas en entrenamiento para evitar sobreajuste
modelo.add(Dense(len(nombres), activation='softmax'))  # capa de salida con tantas clases como personas; softmax convierte salidas en probabilidades para clasificación multicategoria
modelo.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

modelo.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[early_stop],
)


modelo.save("Modelo rostros Clase.keras")
np.save("Clases.npy", label_encoder.classes_)
print("Ya quedo entrenado")