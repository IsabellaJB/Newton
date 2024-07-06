import numpy as np
import math
from numpy.linalg import inv

# ---------------------------------- GOLDEN SEARCH ---------------------------------- 
def regla_eliminacion(x1, x2, fx1, fx2, a, b):
    if fx1 > fx2:
        return x1, b
    if fx1 < fx2:
        return a, x2
    return x1, x2

def w_to_x(w, a, b):
    return w * (b - a) + a

def busquedaDorada(funcion, epsilon, a, b):
    PHI = (1 + math.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    k = 1
    while Lw > epsilon:
        w2 = aw + PHI * Lw
        w1 = bw - PHI * Lw
        aw, bw = regla_eliminacion(w1, w2, funcion(w_to_x(w1, a, b)), funcion(w_to_x(w2, a, b)), aw, bw)
        k += 1
        Lw = bw - aw
    return (w_to_x(aw, a, b) + w_to_x(bw, a, b)) / 2

#  ---------------------------------- FUNCION OBJETIVO ---------------------------------- 
def funcion_objetivo(arreglo):
    x = arreglo[0] 
    y = arreglo[1]
    operacion = ((x**2 + y - 11)**2) + ((x + y**2 - 7)**2)
    return operacion

#  ---------------------------------- GRADIENTE ---------------------------------- 
def gradiente(funcion, x, delta=0.001):
    derivadas = []
    for i in range(len(x)):
        copia = x.copy()
        copia[i] = x[i] + delta
        valor1 = funcion(copia)
        copia[i] = x[i] - delta
        valor2 = funcion(copia)
        derivada = (valor1 - valor2) / (2 * delta)
        derivadas.append(derivada)
    return derivadas

#  ---------------------------------- DISTANCIA ORIGEN ---------------------------------- 
def distancia_origen(vector):
    return np.linalg.norm(vector)

#  ---------------------------------- HESSIANA ---------------------------------- 
def segunda_parte(funcion, x, delta=0.001):
    derivadas2 = []
    for i in range(len(x)):
        copia = x.copy()
        copia[i] = x[i] + delta
        valor1 = funcion(copia)
        valor2 = 2 * (funcion(x))
        copia[i] = x[i] - delta
        valor3 = funcion(copia)
        derivada2 = (valor1 - valor2 + valor3) / (delta**2)
        derivadas2.append(derivada2)
    return derivadas2

def tercera_parte(funcion, x, delta=0.001):
    n = len(x)
    matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                x_ij_plus = x.copy()
                x_ij_plus[i] += delta
                x_ij_plus[j] += delta
                f1 = funcion(x_ij_plus)
                
                x_ij_minus = x.copy()
                x_ij_minus[i] -= delta
                x_ij_minus[j] -= delta
                f2 = funcion(x_ij_minus)
                
                x_i_plus_j_minus = x.copy()
                x_i_plus_j_minus[i] += delta
                x_i_plus_j_minus[j] -= delta
                f3 = funcion(x_i_plus_j_minus)
                
                x_i_minus_j_plus = x.copy()
                x_i_minus_j_plus[i] -= delta
                x_i_minus_j_plus[j] += delta
                f4 = funcion(x_i_minus_j_plus)
                
                derivada2 = (f1 - f3 - f4 + f2) / (4 * delta * delta)
                matriz[i, j] = derivada2
    return matriz

def hessiana(uno, dos):
    n = len(uno)
    matriz = np.zeros((n, n))
    for i in range(n):
        matriz[i, i] = uno[i]
        for j in range(n):
            if i != j:
                matriz[i, j] = dos[i, j]
    return matriz



def newton(funcion_objetivo, x, epsilon1, epsilon2, max_iterations, alpha):
    terminar = False
    xk = np.array(x, dtype=float)
    k = 0
    
    while not terminar:
        # Paso 2: Calcular el gradiente
        gradienteX = np.array(gradiente(funcion_objetivo, xk))
        gradiente_transpuesta = np.transpose(gradienteX)

        # Paso 3: Verificar la condición de terminación
        if np.linalg.norm(gradienteX) < epsilon1 or k >= max_iterations:
            terminar = True
        else:
            # Paso 4: Calcular la hessiana
            h1 = np.array(segunda_parte(funcion_objetivo, xk))
            h2 = np.array(tercera_parte(funcion_objetivo, xk))
            hessian = hessiana(h1, h2)
            hessian = np.matrix(hessian)
            
            # Paso 5: Calcular la inversa de la hessiana
            inversa = inv(hessian)

            # Paso 6: Calcular el producto punto
            punto = np.dot(inversa, gradienteX).A1  # A1 convierte a array 1D

            # Paso 7: Calcular la distancia
            distancia = distancia_origen(gradienteX)

            # Paso 8: Verificar la segunda condición de terminación
            if distancia <= epsilon1 or k >= max_iterations:
                terminar = True
            else:
                # Paso 9: Calcular el nuevo punto x_k1 usando el método de Newton
                def alpha_calcular(alpha):
                    return funcion_objetivo(xk - alpha * punto)
                
                alpha = busquedaDorada(alpha_calcular, epsilon2, 0.0, 1.0)
                
                x_k1 = xk - alpha * punto

                # Paso 10: Verificar la tercera condición de terminación
                if (distancia_origen(x_k1 - xk) / (distancia_origen(xk) + 0.00001)) <= epsilon2:
                    terminar = True
                else:
                    # Paso 11: Actualizar k y xk para la siguiente iteración
                    k = k + 1
                    xk = x_k1

    return xk
# Ejemplo de uso
prueba = [0,0]
epsilon1 = 0.001
epsilon2 = 0.001
max_iterations = 100
alpha = 0.1

punto_final = newton(funcion_objetivo, prueba, epsilon1, epsilon2, max_iterations, alpha)
print("Punto final:", punto_final)
