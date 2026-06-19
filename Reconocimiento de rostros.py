import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Ruta del modelo (archivo .keras) — ajusta si tu archivo tiene otro nombre
MODEL_PATH = "modelo_numeros_clase.keras"
CLASSES_PATH = "clases.npy"  # opcional: archivo con nombres de clases

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

modelo = load_model(MODEL_PATH)

# Cargar nombres de clases si existen, si no, usar etiquetas numéricas
if os.path.exists(CLASSES_PATH):
    clase = np.load(CLASSES_PATH, allow_pickle=True)
else:
    clase = np.array([str(i) for i in range(10)])

ANCHO = 64
ALTO = 64

# Cargamos el clasificador Haar correcto
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascade_path)

camara = cv2.VideoCapture(0)
if not camara.isOpened():
    raise RuntimeError("No se puede abrir la cámara")

print("Cámara iniciada. Presiona 'q' o cierra la ventana para salir.")

# Crear la ventana una sola vez ANTES del bucle
win_name = "Reconocimiento Facial"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

ventana_abierta = True

while ventana_abierta:
    correcto, frame = camara.read()
    if not correcto:
        print("Error al leer frame de la cámara")
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rostros = detector.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=6, minSize=(80, 80))

    for (x, y, w, h) in rostros:
        rostro_color = frame[y:y+h, x:x+w]
        try:
            # Preprocesado: convertir a gris (si el modelo espera 1 canal) y redimensionar
            rostro_gray = cv2.cvtColor(rostro_color, cv2.COLOR_BGR2GRAY)
            rostro_redim = cv2.resize(rostro_gray, (ANCHO, ALTO))
            rostro_normalizado = rostro_redim / 255.0
            entrada = np.expand_dims(rostro_normalizado, axis=(0, -1)).astype(np.float32)  # (1, ALTO, ANCHO, 1)

            prediccion = modelo.predict(entrada, verbose=0)
            indice = int(np.argmax(prediccion))
            nombre = clase[indice] if indice < len(clase) else str(indice)
            confianza = float(np.max(prediccion))
            color = (0, 255, 0) if confianza > 0.7 else (0, 0, 255)
            texto = f"{nombre} ({confianza * 100:.1f}%)"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except Exception as e:
            cv2.putText(frame, "Error rostro", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0))
            print("Error al procesar rostro:", e)

    cv2.imshow(win_name, frame)

    # Salir con 'q' o cuando la ventana se cierre
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'):
        print("Cerrado por el usuario (tecla 'q')")
        ventana_abierta = False
    
    # Detectar si la ventana se cerró manualmente
    try:
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Ventana cerrada por el usuario. Saliendo...")
            ventana_abierta = False
    except cv2.error:
        # Si la ventana no existe o hubo error, salir
        ventana_abierta = False

camara.release()
cv2.destroyAllWindows()
