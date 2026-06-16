import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten, Dense, Dropout

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator


RUTA_DATOS = r"/home/carlosorozco/Escritorio/Sistemas inteligentes/Unidad 2/Reconocimiento de rostros"

ANCHO = 256
ALTO = 256

imagenes = []
etiquetas = []
nombres = []

for nombre_persona in os.listdir(RUTA_DATOS):
    ruta_persona = os.path.join(RUTA_DATOS , nombre_persona)
    if os.path.isdir(ruta_persona):
        nombres.append(nombre_persona)
        for archivo_imagen in os.listdir (ruta_persona):
            ruta_imagen = os.path.join (ruta_persona, archivo_imagen)
            imagen = cv2.imread(ruta_imagen)
            if imagen is not None:
                imagen = cv2.resize(imagen, (ANCHO,ALTO))
                imagen = imagen / 255.0
                imagen.append(imagen)
                etiquetas.append(nombre_persona)
imagenes = np.array (imagenes)
etiquetas = np.array (etiquetas)

label_encoder = LabelEncoder()
etiquetas_numericas = label_encoder.fit_transform
(etiquetas)
