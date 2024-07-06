import numpy as np



# OBJETIVO --------------------------------------
def himmelblau(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2



# FIJA ------------------------------------------
def gradient(x):
    x1, x2 = x
    df_dx1 = 4*x1*(x1**2 + x2 - 11) + 2*(x1 + x2**2 - 7)
    df_dx2 = 2*(x1**2 + x2 - 11) + 4*x2*(x1 + x2**2 - 7)
    return np.array([df_dx1, df_dx2])

def hessian(x):
    x1, x2 = x
    d2f_dx1_dx1 = 12*x1**2 + 4*x2 - 42
    d2f_dx1_dx2 = 4*(x1 + x2)
    d2f_dx2_dx2 = 4*x1 + 12*x2**2 - 26
    return np.array([[d2f_dx1_dx1, d2f_dx1_dx2], 
                     [d2f_dx1_dx2, d2f_dx2_dx2]])















# Método de Newton
def newton_method(x0, tol=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad = gradient(x)
        hess = hessian(x)
        try:
            hess_inv = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            print("La matriz Hessiana no es invertible.")
            return None
        
        step = -hess_inv @ grad
        x_new = x + step
        
        if np.linalg.norm(step) < tol:
            print(f"Convergencia alcanzada en {i+1} iteraciones.")
            return x_new
        
        x = x_new
        print(f"Iteración {i+1}: x = {x}, f(x) = {himmelblau(x)}")
    
    print("El método no convergió.")
    return x

# Valores iniciales
x0_1 = [0, 0]
x0_2 = [2, 1]


print("Prueba con el valor inicial [0, 0]:")
x_min_1 = newton_method(x0_1)
print(f"El mínimo encontrado es: x = {x_min_1}, f(x) = {himmelblau(x_min_1)}")


print("\nPrueba con el valor inicial [2, 1]:")
x_min_2 = newton_method(x0_2)
print(f"El mínimo encontrado es: x = {x_min_2}, f(x) = {himmelblau(x_min_2)}")
