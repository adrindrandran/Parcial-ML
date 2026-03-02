import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import SparseCoder

# 1. Definimos el Diccionario (Átomos)
# Cada fila es un "átomo" o patrón base.
dictionary = np.array([
    [1, 1, 1, 0, 0, 0], # Patrón A: Activación al inicio
    [0, 0, 0, 1, 1, 1], # Patrón B: Activación al final
    [1, 0, 1, 0, 1, 0], # Patrón C: Activación alternada
    [0, 1, 0, 1, 0, 1]  # Patrón D: Activación alternada inversa
], dtype=np.float64)

# 2. Creamos una señal (Data) que queremos representar
# Esta señal se parece un poco al Patrón A y mucho al Patrón B
y = np.array([0.5, 0.5, 0.5, 2.0, 2.0, 2.0])

# 3. Inicializamos el SparseCoder
# Usamos 'lasso_lars' que es un algoritmo eficiente para encontrar la dispersión
coder = SparseCoder(dictionary=dictionary, transform_algorithm='lasso_lars', transform_alpha=0.1)

# 4. Ejecutamos la codificación
sparse_code = coder.transform(y.reshape(1, -1))

print("Código disperso (coeficientes):", sparse_code)