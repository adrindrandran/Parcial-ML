import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1) IMPORTACIÓN Y PREPARACIÓN DEL DATASET
# ==========================================
url = "https://raw.githubusercontent.com/vega/vega/main/docs/data/seattle-weather.csv"

try:
    data = pd.read_csv(url)
    features = ['precipitation', 'temp_max', 'temp_min', 'wind']
    # Eliminamos filas con nulos y nos quedamos con las columnas numéricas
    X_raw = data[features].dropna()
    print(f" [OK] Dataset cargado: {X_raw.shape[0]} registros meteorológicos encontrados.\n")
except Exception as e:
    print(f" [ERROR] No se pudo acceder a los datos: {e}")
    exit()

# Escalado: Fundamental para que unidades distintas (mm vs °C) sean comparables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# =========================================================
# 2) APRENDIZAJE DE DICCIONARIO (DICTIONARY LEARNING)
# =========================================================
# n_components=4: Definimos 4 "estados climáticos" fundamentales (átomos)
# alpha=1: Controla la escasez (sparsity). A mayor alpha, más ceros en la matriz.
n_atoms = 4
dict_learner = DictionaryLearning(
    n_components=n_atoms, 
    alpha=1.0, 
    transform_algorithm='lasso_lars', 
    random_state=42,
    max_iter=1000
)

# sparse_representation (H): Cómo se compone cada día (coeficientes)
# dictionary_atoms (D): Qué características definen cada patrón
sparse_representation = dict_learner.fit_transform(X_scaled)
dictionary_atoms = dict_learner.components_

# =========================================================
# 3) INTERPRETACIÓN Y ETIQUETADO DE RESULTADOS
# =========================================================
# Revertimos el escalado para entender los átomos en unidades reales
interpreted_atoms = scaler.inverse_transform(dictionary_atoms)
df_atoms = pd.DataFrame(interpreted_atoms, columns=features)

def asignar_nombre_meteorologico(row):
    """Asigna un nombre lógico basado en los valores físicos del átomo."""
    labels = []
    if row['precipitation'] > 3: labels.append("Lluvioso/Húmedo")
    if row['temp_max'] > 18: labels.append("Cálido/Soleado")
    if row['temp_min'] < 6: labels.append("Frío/Invernal")
    if row['wind'] > 4.2: labels.append("Ventoso")
    
    return " & ".join(labels) if labels else "Estable/Despejado"

nombres_logicos = [f"Átomo {i}: {asignar_nombre_meteorologico(row)}" for i, row in df_atoms.iterrows()]
df_atoms.index = nombres_logicos

print("="*80)
print(" DICCIONARIO DE PATRONES APRENDIDOS (Significado Físico)")
print("="*80)
print(df_atoms.round(2))
print("\n" + "-"*80)

# =========================================================
# 4) VISUALIZACIÓN DE RESULTADOS
# =========================================================
# Gráfico 1: Heatmap de los Átomos
plt.figure(figsize=(12, 6))
sns.heatmap(df_atoms, annot=True, cmap="YlOrRd", fmt=".2f", cbar_kws={'label': 'Valor Real'})
plt.title("Mapa de Calor de los Átomos Aprendidos\n(Perfil físico de cada patrón climático)", fontsize=14)
plt.show()

# Gráfico 2: Esparcidad de la Matriz de Coeficientes
plt.figure(figsize=(12, 4))
plt.spy(sparse_representation[:150].T, precision=0.01, aspect='auto', marker='s', markersize=4, color='teal')
plt.title("Visualización de la Matriz de Coeficientes (Esparcidad)\nCada punto es un patrón activo en un día específico", fontsize=12)
plt.ylabel("Índice del Átomo")
plt.xlabel("Días (Primeros 150)")
plt.show()

# =========================================================
# 5) MATRIZ DE COEFICIENTES (H) COMPLETA (Muestra)
# =========================================================
print("\n" + "="*80)
print(" MATRIZ DE COEFICIENTES (H) - REPRESENTACIÓN ESPARSA (Primeros 15 días)")
print("="*80)
# Creamos un DataFrame para que sea legible
h_matrix_display = pd.DataFrame(sparse_representation, columns=[f"A{i}" for i in range(n_atoms)])
print(h_matrix_display.head(15).round(4))

# Cálculo de Sparsity Final
sparsity_pct = np.mean(sparse_representation == 0) * 100
print(f"\n [INFO] Nivel de esparcidad: {sparsity_pct:.2f}% de la matriz son ceros.")
print(" [INFO] Interpretación: Cada día se explica mediante una combinación simple de estos átomos.")