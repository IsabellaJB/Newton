
import numpy as np
import math
import matplotlib.pyplot as plt
from abc import ABC
from numpy.linalg import det, inv




# ---------------------------------- GOLDEN SEARCH ---------------------------------- 
def regla_eliminacion(x1, x2, fx1, fx2, a, b):
    if fx1 > fx2:
        return x1, b
    if fx1 < fx2:
        return a, x2

    return x1, x2

def w_to_x(w, a, b):
    return w*(b-a) + a

def busquedaDorada(funcion, epsilon, a, b):
    PHI = (1+math.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    k = 1
    while Lw > epsilon:
        w2 = aw + PHI*Lw
        w1 = bw - PHI*Lw
        aw, bw = regla_eliminacion(w1,w2, funcion(w_to_x(w1,a,b)), funcion(w_to_x(w2,a,b)), aw, bw)
        k+=1
        Lw = bw - aw
    return (w_to_x(aw, a, b)+w_to_x(bw,a,b))/2




#  ---------------------------------- FUNCION OBJETIVO ---------------------------------- 
def funcion_objetivo(arreglo):
    x = arreglo[0] 
    y = arreglo[1]
    operacion = ((x**2 + y - 11)**2) + ((x + y**2 - 7)**2)
    return operacion




#  ---------------------------------- GRADIENTE ---------------------------------- 
def gradiente(funcion, x, delta=0.001):
    derivadas = []
    for i in range (0, len(x)):
        valor1 = 0
        valor2 = 0
        valor_final = 0
        copia = x.copy()
        copia[i] = x[i] + delta
        valor1 = funcion(copia)
        copia[i] = x[i] - delta
        valor2 = funcion(copia)
        valor_final = (valor1 - valor2) / (2*delta)
        derivadas.append(valor_final)
    return derivadas




#  ---------------------------------- DISTANCIA ORIGEN ---------------------------------- 
def distancia_origen(vector):
    return np.linalg.norm(vector)




#  ---------------------------------- REDONDEAR ---------------------------------- 
def redondear(arreglo):
    lita = []
    for valor in arreglo:
        v = round(valor, 2)
        lita.append(v)
    return(lita)




#  ---------------------------------- HESSIANA ---------------------------------- 
def segunda_parte(funcion, x, delta=0.001):
    derivadas2 = []
    lista2 = []
    for i in range (0, len(x)):
        valor1 = 0
        valor2 = 0
        valor3 = 0
        valor_final = 0
        copia = x.copy()
        copia[i] = x[i] + delta
        valor1 = funcion(copia)
        valor2 = 2*(funcion(x))
        copia[i] = x[i] - delta
        valor3 = funcion(copia)
        valor_final = (valor1 - (valor2) + valor3) / (delta**2)
        lista2.append(valor_final)
    return lista2


def tercera_parte(funcion, x, delta=0.001):
    lista_final = []
    for i in range(len(x)):
        lista = []
        lista2 = []
        lista3 = []
        lista4 = []
        for j in range(len(x)):
            parte_uno_uno = (x[i]+delta)
            parte_uno_dos = (x[j]+delta)
            lista.append(parte_uno_uno)
            lista.append(parte_uno_dos)
            v1 = funcion(lista)
            parte_dos_uno = (x[i] + delta)
            parte_dos_dos = (x[j] - (delta))
            lista2.append(parte_dos_uno)
            lista2.append(parte_dos_dos)
            v2 = funcion(lista2)
            parte_tres_uno = (x[i]-delta)
            parte_tres_dos = (x[j]+delta)
            lista3.append(parte_tres_uno)
            lista3.append(parte_tres_dos)
            v3 = funcion(lista3)
            parte_cuatro_uno = (x[i]-delta)
            parte_cuatro_dos = (x[j]-delta)
            lista4.append(parte_cuatro_uno)
            lista4.append(parte_cuatro_dos)
            v4 = funcion(lista4)
        v_f = (v1 - (v2) - (v3) + v4) / (4*delta*delta)
        lista_final.append(v_f)
    return lista_final
        


def hessiana(uno, dos):
    n = len(uno)
    matriz = [[0] * n for _ in range(n)]
    for i in range(n):
        matriz[i][i] = uno[i]
    for i in range(n):
        matriz[i][n - 1 - i] = dos[i]
    return matriz




prueba = [1,1]
deltaX = 0.01        


gradiente_X = gradiente(funcion_objetivo,prueba)

uno = (segunda_parte(funcion_objetivo,prueba))
dos = (tercera_parte(funcion_objetivo,prueba))


print("Gradiente: {}".format(gradiente))
print("Hessiana:")
hessian = hessiana(uno,dos)
# print(hessian)

hessian = np.matrix(hessian)
print(hessian)

inversa = inv(hessian)
print(inversa)

punto = np.dot(hessian, gradiente_X).A1
print(punto)


# ope = [0,0] - 0.2*(punto)
# print(ope)



def newton(funcion, funcion_objetivo, x, epsilon1, epsilon2, max_iterations, alpha):
    terminar = False
    xk = x
    k = 0
    while not terminar:
        # GRADIENTE
        gradienteX = np.array(gradiente(funcion_objetivo, xk))

        # HESSIANA
        h1 = (segunda_parte(funcion_objetivo, xk))
        h2 = (tercera_parte(funcion_objetivo, xk))
        hessian = hessiana(h1,h2)
        hessian = np.matrix(hessian)

        # INVERSA
        inversa = inv(hessian)

        # PRODUCTO PUNTO
        punto = np.dot(inversa, gradienteX)

        # DISTANCIA
        distancia = distancia_origen(gradienteX)
        if distancia <= epsilon1:
            terminar = True
        elif (k >= max_iterations):
            terminar = True
        else:
            # --------------------------------------------------
            # ----------------- PASO 4 -------------------------



            def alpha_calcular(alpha):
                # punto = np.dot(inversa, gradienteX)
                return funcion_objetivo(xk - alpha * punto)
            
            alpha = funcion(alpha_calcular,epsilon2, 0.0,1.0)
            # alpha = funcion(alpha_calcular,funcion_objetivo)
            
            x_k1 = xk - alpha * punto
            # print(xk,alpha, gradienteX, x_k1)




            # --------------------------------------------------


            if (distancia_origen(x_k1-xk)/distancia_origen(xk)+0.00001) <= epsilon2:
                terminar = True
            else:
                k = k + 1
                xk = x_k1
    return xk

    


max_iterations = 100
x = [0.0,0.0]
deltaX = 0.01

epsilon1 = 0.001
epsilon2 = 0.001
k = 0
alpha = 0.2


punto_final = (newton(busquedaDorada, funcion_objetivo, x, epsilon1, epsilon2, max_iterations, alpha))
print(punto_final)

# nuevos = redondear(punto_final)
# print(nuevos)

