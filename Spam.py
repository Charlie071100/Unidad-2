import numpy as np

# 50 ejemplos de mensajes que NO son spam (etiqueta 0)
ham_texts = [
    "hola",
    "buenos dias",
    "buenas tardes",
    "que tal",
    "gracias por tu ayuda",
    "tengo una pregunta",
    "nos vemos mañana",
    "recordatorio de la clase",
    "este es un mensaje normal",
    "adjunto el informe",
    "reunión el viernes",
    "cita con el doctor",
    "confirmación de asistencia",
    "el evento es en la escuela",
    "por favor revisa el archivo",
    "feliz cumpleaños",
    "te escribo por el proyecto",
    "avísame si puedes venir",
    "información de la tarea",
    "te mando un saludo",
    "¿cómo estás?",
    "nos juntamos hoy",
    "el examen será pronto",
    "estoy en el trabajo",
    "hablamos en la tarde",
    "te llamo más tarde",
    "la reunión fue cancelada",
    "necesito más detalles",
    "envíame tu dirección",
    "el viaje será en junio",
    "tu pedido está listo",
    "la entrega llega mañana",
    "la clase empieza a las ocho",
    "el documento está completo",
    "tengo dudas sobre la tarea",
    "quiero practicar juntos",
    "felicidades por tu logro",
    "nos vemos en la biblioteca",
    "saludos cordiales",
    "gracias por la invitación",
    "el taller es gratuito",
    "habrá comida en la reunión",
    "las notas ya están disponibles",
    "te paso el enlace",
    "por favor contesta esta encuesta",
    "el curso es para todos",
    "lleva lápiz y cuaderno",
    "estoy enviando esta invitación",
    "buenos deseos para ti",
    "nos vemos en el campus",
    "es un mensaje importante pero normal"
]

# 50 ejemplos de mensajes de SPAM (etiqueta 1)
spam_texts = [
    "te ganaste un iphone",
    "fuiste seleccionado para una rifa",
    "gana dinero rapido",
    "haz clic aqui",
    "necesitas actualizar tu cuenta",
    "reclama tu premio ahora",
    "oferta especial por tiempo limitado",
    "premio gratis disponible",
    "tu cuenta ha sido bloqueada",
    "verifica tu informacion",
    "recibe dinero facil",
    "compra ahora y ahorra",
    "tu tarjeta fue aprobada",
    "obtén un descuento urgente",
    "ganas un viaje gratis",
    "haz clic para reclamar",
    "sin costo por tiempo limitado",
    "premio instantaneo aqui",
    "tu saldo ha aumentado",
    "eres el ganador",
    "has sido seleccionado",
    "obtén regalos gratuitos",
    "tu paquete está esperando",
    "haz clic para actualizar",
    "falta información urgente",
    "gana dinero desde casa",
    "consulta tu premio ahora",
    "regalo gratis solo hoy",
    "aumenta tus ingresos rápido",
    "obtén criptomonedas gratis",
    "tu cuenta necesita verificación",
    "verifica tu pago",
    "haz clic para cobrar",
    "recibe regalos en segundos",
    "actualiza tu cuenta bancaria",
    "aquí tienes un cupón gratis",
    "pagos diarios garantizados",
    "aviso urgente sobre tu cuenta",
    "alerta de seguridad falsa",
    "te damos un iphone gratis",
    "se seleccionó tu teléfono",
    "cobras sin trabajar",
    "promoción urgente para ti",
    "tu correo ganó un premio",
    "ahora puedes ganar dinero",
    "haz clic para ganar",
    "ganas un auto gratis",
    "oferta limitada de regalo",
    "recibe un cheque gratis",
    "responde para cobrar",
    "ganador de la rifa aquí"
]

texts = ham_texts + spam_texts
labels = [0] * len(ham_texts) + [1] * len(spam_texts)

# Unimos las dos bases de datos en una sola lista.
datos = texts

# bag of words: juntamos todas las frases para obtener el vocabulario.
corpus = " ".join(datos).lower()
corpus = corpus.replace("?", "").replace(",", "").replace(".", "")

vocabulario = sorted(set(corpus.split()))


def texto_a_vector(texto):
    texto = texto.lower()
    texto = texto.replace("?", "").replace(",", "").replace(".", "")
    palabras_texto = texto.split()

    vector = np.zeros(len(vocabulario), dtype=np.float32)
    for i, palabra in enumerate(vocabulario):
        if palabra in palabras_texto:
            vector[i] = 1.0
    return vector


X = np.array([texto_a_vector(t) for t in datos], dtype=np.float32)
y = np.array(labels, dtype=np.float32).reshape(-1, 1)

np.random.seed(1)
input_size = X.shape[1]
hidden_size = 10
output_size = 1

w1 = np.random.rand(input_size, hidden_size) - 0.5
b1 = np.zeros((1, hidden_size))
w2 = np.random.rand(hidden_size, output_size) - 0.5
b2 = np.zeros((1, output_size))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def entrenar(X, y, epochs=1000, lr=0.2):
    global w1, b1, w2, b2
    for epoch in range(1, epochs + 1):
        z1 = np.dot(X, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)

        error = y - a2
        d2 = error * sigmoid_derivative(a2)
        d1 = np.dot(d2, w2.T) * sigmoid_derivative(a1)

        w2 += lr * np.dot(a1.T, d2)
        b2 += lr * np.sum(d2, axis=0, keepdims=True)
        w1 += lr * np.dot(X.T, d1)
        b1 += lr * np.sum(d1, axis=0, keepdims=True)

        if epoch % 200 == 0 or epoch == 1:
            loss = np.mean(np.square(error))
            print(f"Época {epoch}/{epochs} - pérdida: {loss:.4f}")


def predecir(texto):
    x = np.array([texto_a_vector(texto)], dtype=np.float32)
    a1 = sigmoid(np.dot(x, w1) + b1)
    a2 = sigmoid(np.dot(a1, w2) + b2)
    score = float(a2[0, 0])
    etiqueta = "SPAM" if score > 0.5 else "NO SPAM"
    return etiqueta, score


print("Entrenando la red neuronal para detectar spam...")
entrenar(X, y, epochs=1000, lr=0.2)

print("\nAhora prueba con un mensaje del profe:")
mensaje_profesor = input("Escribe el mensaje: ")
etiqueta, score = predecir(mensaje_profesor)
print(f"Resultado: {etiqueta} ({score:.3f})")
