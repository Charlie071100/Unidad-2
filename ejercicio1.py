import numpy as np

x= np.array([[0,0],[0,1], [1,0],[1,1]])
y = np.array([0,0,0,1])

w= np.random.rand(2)
b = np.random.rand(1)
lr = 0.2 #tasa de aprendizaje

print("Pesos iniciales: ",w,y,b)

for epoch in range (50):
    for i in range (len(x)):
        z =  np.dot(x[i],w) + b


        if z > 0.5:
            y_pred = 1
        else:
            y_pred = 0
        
        error = y[1] - y_pred

        w += lr * error * x[i]
        b += lr * error

        print("Pesos Finales",w,"Bias: ",b)

    for i in range (len (x)):
        z = np.dot (x[i],w) + b
        if z> 0.5:
            y_pred = 1
        else:
            y_pred = 0
        print (f"Entrada: {x[i]} ||| Salida: {y_pred}")

    def predecir (x):
        z =  np.dot (x,w) + b
        if z > 0.5:
            y_pred = 1
        else: 
            y_pred = 0
        return y_pred

    nuevo = np.array ([1,1])
    print (f"Ingresaste {nuevo} y su resultado es {predecir(nuevo)}")