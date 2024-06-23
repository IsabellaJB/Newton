import numpy as np
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from numpy.linalg import inv

def funcion_objetivo(x, y):
    operacion = ((x**2 + y - 11)**2) + ((x + y**2 - 7)**2)
    return operacion

class Optimizador(ABC):
    def __init__(self, funcion_objetivo):
        self.funcion_objetivo = funcion_objetivo

    @abstractmethod
    def optimizar(self, *args):
        pass

class Newton(Optimizador):
    def __init__(self, x, funcion_objetivo, epsilon1, epsilon2, M, metodo_univariable):
        super().__init__(funcion_objetivo)
        self.x = x
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.M = M
        self.metodo_univariable = metodo_univariable

    def gradiente(self, x, delta=0.001):
        derivadas = []
        for i in range(len(x)):
            copia = x.copy()
            copia[i] = x[i] + delta
            valor1 = self.funcion_objetivo(*copia)
            copia[i] = x[i] - delta
            valor2 = self.funcion_objetivo(*copia)
            valor_final = (valor1 - valor2) / (2 * delta)
            derivadas.append(valor_final)
        return derivadas

    def distancia_origen(self, vector):
        return np.linalg.norm(vector)

    def segunda_parte(self, x, delta=0.001):
        derivadas2 = []
        for i in range(len(x)):
            copia = x.copy()
            copia[i] = x[i] + delta
            valor1 = self.funcion_objetivo(*copia)
            valor2 = 2 * (self.funcion_objetivo(*x))
            copia[i] = x[i] - delta
            valor3 = self.funcion_objetivo(*copia)
            derivada2 = (valor1 - valor2 + valor3) / (delta**2)
            derivadas2.append(derivada2)
        return derivadas2

    def tercera_parte(self, x, delta=0.001):
        n = len(x)
        matriz = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    x_ij_plus = x.copy()
                    x_ij_plus[i] += delta
                    x_ij_plus[j] += delta
                    f1 = self.funcion_objetivo(*x_ij_plus)
                    
                    x_ij_minus = x.copy()
                    x_ij_minus[i] -= delta
                    x_ij_minus[j] -= delta
                    f2 = self.funcion_objetivo(*x_ij_minus)
                    
                    x_i_plus_j_minus = x.copy()
                    x_i_plus_j_minus[i] += delta
                    x_i_plus_j_minus[j] -= delta
                    f3 = self.funcion_objetivo(*x_i_plus_j_minus)
                    
                    x_i_minus_j_plus = x.copy()
                    x_i_minus_j_plus[i] -= delta
                    x_i_minus_j_plus[j] += delta
                    f4 = self.funcion_objetivo(*x_i_minus_j_plus)
                    
                    derivada2 = (f1 - f3 - f4 + f2) / (4 * delta * delta)
                    matriz[i, j] = derivada2
        return matriz

    def hessiana(self, h1, h2):
        n = len(h1)
        matriz = np.zeros((n, n))
        for i in range(n):
            matriz[i, i] = h1[i]
            for j in range(n):
                if i != j:
                    matriz[i, j] = h2[i, j]
        return matriz

    def optimizar(self):
        terminar = False
        xk = np.array(self.x, dtype=float)
        k = 0
        while not terminar:
            gradienteX = np.array(self.gradiente(xk))
            h1 = np.array(self.segunda_parte(xk))
            h2 = np.array(self.tercera_parte(xk))
            hessian = self.hessiana(h1, h2)
            inversa = inv(hessian)
            punto = np.dot(inversa, gradienteX).A1

            distancia = self.distancia_origen(gradienteX)
            if distancia <= self.epsilon1:
                terminar = True
            elif k >= self.M:
                terminar = True
            else:
                def alpha_calcular(alpha):
                    return self.funcion_objetivo(*(xk - alpha * punto))

                alpha = self.metodo_univariable.optimizar(alpha_calcular, 0.0, 1.0)
                x_k1 = xk - alpha * punto

                if (self.distancia_origen(x_k1 - xk) / (self.distancia_origen(xk) + 0.00001)) <= self.epsilon2:
                    terminar = True
                else:
                    k += 1
                    xk = x_k1

        return xk

class GoldenSearch(Optimizador):
    def __init__(self, funcion_objetivo, epsilon, a, b):
        super().__init__(funcion_objetivo)
        self.epsilon = epsilon
        self.a = a
        self.b = b

    def regla_eliminacion(self, x1, x2, fx1, fx2, a, b):
        if fx1 > fx2:
            return x1, b
        if fx1 < fx2:
            return a, x2
        return x1, x2

    def w_to_x(self, w, a, b):
        return w * (b - a) + a

    def optimizar(self, funcion, a, b):
        PHI = (1 + math.sqrt(5)) / 2 - 1
        aw, bw = 0, 1
        Lw = 1
        while Lw > self.epsilon:
            w2 = aw + PHI * Lw
            w1 = bw - PHI * Lw
            aw, bw = self.regla_eliminacion(
                w1, w2,
                funcion(self.w_to_x(w1, a, b)),
                funcion(self.w_to_x(w2, a, b)),
                aw, bw
            )
            Lw = bw - aw
        return (self.w_to_x(aw, a, b) + self.w_to_x(bw, a, b)) / 2

class Fibonacci(Optimizador):
    def __init__(self, funcion_objetivo, epsilon):
        super().__init__(funcion_objetivo)
        self.epsilon = epsilon

    def fibonacci(self, n):
        if n <= 1:
            return n
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b

    def optimizar(self, funcion, a, b):
        n = 0
        while self.fibonacci(n) < (b - a) / self.epsilon:
            n += 1
        L = b - a
        for k in range(1, n):
            L_k = self.fibonacci(n - k) / self.fibonacci(n) * L
            x1 = a + L_k
            x2 = b - L_k
            if funcion(x1) > funcion(x2):
                a = x1
            else:
                b = x2
        return (a + b) / 2

prueba = [3, 2]

# Golden Search con Newton
opt_golden = GoldenSearch(funcion_objetivo, epsilon=0.001, a=0.1, b=0.0)
opt_newton_golden = Newton(prueba, funcion_objetivo, epsilon1=0.001, epsilon2=0.001, M=100, metodo_univariable=opt_golden)
resultado_newton_golden = opt_newton_golden.optimizar()
print("Resultado Newton con Golden Search:", resultado_newton_golden)

# Fibonacci con Newton
opt_fibonacci = Fibonacci(funcion_objetivo, epsilon=0.001)
opt_newton_fibonacci = Newton(prueba, funcion_objetivo, epsilon1=0.001, epsilon2=0.001, M=100, metodo_univariable=opt_fibonacci)
resultado_newton_fibonacci = opt_newton_fibonacci.optimizar()
print("Resultado Newton con Fibonacci:", resultado_newton_fibonacci)
