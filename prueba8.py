import numpy as np
import math

# ---------------------------------- FUNCION OBJETIVO ---------------------------------- 
def funcion_objetivo(arreglo):
    x = arreglo[0]
    y = arreglo[1]
    operacion = ((x**2 + y - 11)**2) + ((x + y**2 - 7)**2)
    return operacion

# ---------------------------------- GRADIENTE ---------------------------------- 
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
    return np.array(derivadas)

# ---------------------------------- HESSIANA ---------------------------------- 
def hessiana(funcion, x, delta=0.001):
    n = len(x)
    matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                copia1 = x.copy()
                copia1[i] += delta
                f1 = funcion(copia1)
                
                copia2 = x.copy()
                copia2[i] -= delta
                f2 = funcion(copia2)
                
                matriz[i, i] = (f1 - 2 * funcion(x) + f2) / (delta**2)
            elif i < j:
                copia3 = x.copy()
                copia3[i] += delta
                copia3[j] += delta
                f3 = funcion(copia3)
                
                copia4 = x.copy()
                copia4[i] += delta
                copia4[j] -= delta
                f4 = funcion(copia4)
                
                copia5 = x.copy()
                copia5[i] -= delta
                copia5[j] += delta
                f5 = funcion(copia5)
                
                copia6 = x.copy()
                copia6[i] -= delta
                copia6[j] -= delta
                f6 = funcion(copia6)
                
                matriz[i, j] = (f3 - f4 - f5 + f6) / (4 * delta * delta)
                matriz[j, i] = matriz[i, j]
    return matriz

# ---------------------------------- BUSQUEDA DORADA ---------------------------------- 
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

# ---------------------------------- DISTANCIA ORIGEN ---------------------------------- 
def distancia_origen(vector):
    return np.linalg.norm(vector)

# ---------------------------------- MÉTODO DE NEWTON MODIFICADO ---------------------------------- 
def newton(funcion_objetivo, x0, epsilon1=1e-6, epsilon2=1e-6, max_iterations=100):
    terminar = False
    xk = np.array(x0, dtype=float)
    k = 0
    while not terminar:
        # GRADIENTE
        gradienteX = gradiente(funcion_objetivo, xk)
        
        if np.linalg.norm(gradienteX) < epsilon1 or k >= max_iterations:
            terminar = True
        else:
            # HESSIANA
            hessian = hessiana(funcion_objetivo, xk)
            try:
                inversa = np.linalg.inv(hessian)
            except np.linalg.LinAlgError:
                print("La matriz Hessiana no es invertible.")
                return None
            
            # PRODUCTO PUNTO
            punto = inversa @ gradienteX
            
            # Búsqueda de línea usando búsqueda dorada
            def alpha_calcular(alpha):
                return funcion_objetivo(xk - alpha * punto)
            
            alpha = busquedaDorada(alpha_calcular, epsilon2, 0.0, 1.0)
            
            x_k1 = xk - alpha * punto

            if (distancia_origen(x_k1 - xk) / (distancia_origen(xk) + 0.00001)) <= epsilon2:
                terminar = True
            else:
                k += 1
                xk = x_k1
        
        print(f"Iteración {k+1}: x = {xk}, f(x) = {funcion_objetivo(xk)}")

    if k < max_iterations:
        print(f"Convergencia alcanzada en {k+1} iteraciones.")
    else:
        print("El método no convergió.")
    
    return xk


x0 = [2,1]
solucion = newton(funcion_objetivo, x0)
print("Solución encontrada:", solucion)
