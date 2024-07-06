import numpy as np
import math

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

# ---------------------------------- MÉTODO DE NEWTON MODIFICADO ---------------------------------- 
def newton_method(funcion, x0, tol=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad = gradiente(funcion, x)
        hess = hessiana(funcion, x)
        try:
            hess_inv = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            print("La matriz Hessiana no es invertible.")
            return None
        
        step = -hess_inv @ grad
        x_new = x + step
        
        if np.linalg.norm(step) < tol:
            print(f"Convergencia alcanzada en {i+1} iteraciones.")
            print(f"Solución encontrada: x = {x_new}, f(x) = {funcion(x_new)}")
            return x_new
        
        x = x_new
        print(f"Iteración {i+1}: x = {x}, f(x) = {funcion(x)}")
    
    print("El método no convergió.")
    print(f"Mejor solución encontrada: x = {x}, f(x) = {funcion(x)}")
    return x

print(newton_method(funcion_objetivo,[2,1]))