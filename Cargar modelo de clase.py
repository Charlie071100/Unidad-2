import cv2, numpy as np
from tensorflow.keras.models import load_model

modelo = load_model("modelo_numeros_clase.keras")


IMAGEN_ANCHO = 64
IMAGEN_ALTO = 64

camara = cv2.VideoCapture(0)
if not camara.isOpened():
    raise RuntimeError("No se puede abrir la cámara")

while True:
    correcto, frame = camara.read()
    if not correcto:
        break

    x_inicio = 100
    y_inicio = 100
    ancho = 200
    alto = 200
    cv2.rectangle(frame, (x_inicio, y_inicio), (x_inicio+ancho, y_inicio+alto), (255,0,0), 2)

    recorte = frame[y_inicio:y_inicio+alto, x_inicio:x_inicio+ancho]
    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    redimensionando = cv2.resize(gris, (IMAGEN_ANCHO, IMAGEN_ALTO))
    normalizando = redimensionando / 255.0
    entrada = normalizando.reshape(1, IMAGEN_ALTO, IMAGEN_ANCHO, 1).astype(np.float32)

    prediccion = modelo.predict(entrada)
    numero = np.argmax(prediccion)

    texto = f"Numero: {numero}"
    cv2.putText(frame, texto, (x_inicio, y_inicio-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Reconocimiento de numero", frame)

    if cv2.waitKey(1) == 27:
        break

camara.release()
cv2.destroyAllWindows()