import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# -------------------------------------------------------------------
# 1. CARGA DE DATOS (50 imágenes, sin fondo, escala de grises, tamaño nativo)
# -------------------------------------------------------------------
print("Cargando 50 imágenes de LFW a resolución nativa...")
lfw = fetch_lfw_people(min_faces_per_person=20, resize=None, color=False)
imagenes = lfw.images[:50]  

patch_size = (10, 10)
print("Extrayendo parches de las 50 imágenes...")
parches_lista = [extract_patches_2d(img, patch_size) for img in imagenes]
parches = np.vstack(parches_lista)
parches_flat = parches.reshape(parches.shape[0], -1)

# Normalización estricta
scaler = StandardScaler()
parches_norm = scaler.fit_transform(parches_flat)

print(f"Total de parches a procesar: {len(parches_norm)}")
print("-" * 60)

# -------------------------------------------------------------------
# 2. ESTUDIO PARAMÉTRICO EXHAUSTIVO (De 4 a 100 átomos)
# -------------------------------------------------------------------
rango_atomos = range(4, 101)  # De 4 hasta 100 inclusive
resultados_mse = []
resultados_esparcidad = []

tiempo_inicio_total = time.time()

print("Iniciando barrido de diccionarios. Esto tomará su tiempo...\n")

for n_atomos in rango_atomos:
    t0_iteracion = time.time()
    
    # Entrenamiento forzando todos los núcleos del Mac (n_jobs=-1)
    dico = MiniBatchDictionaryLearning(
        n_components=n_atomos, 
        alpha=1.0, 
        max_iter=50,  # Reducido un poco para aligerar el bucle gigante
        batch_size=256, 
        random_state=42,
        n_jobs=-1 
    )
    
    # Aprender diccionario y obtener coeficientes a la vez
    codigos = dico.fit_transform(parches_norm)
    
    # Cálculo exacto de esparcidad (porcentaje de ceros)
    ceros_exactos = np.sum(np.abs(codigos) < 1e-5)
    esparcidad_pct = (ceros_exactos / codigos.size) * 100
    resultados_esparcidad.append(esparcidad_pct)
    
    # Reconstrucción matemática pura
    parches_rec_norm = np.dot(codigos, dico.components_)
    
    # Cálculo del Error Cuadrático Medio (MSE exacto, sin raíces)
    mse_exacto = mean_squared_error(parches_norm, parches_rec_norm)
    resultados_mse.append(mse_exacto)
    
    tiempo_iteracion = time.time() - t0_iteracion
    
    # Imprimir progreso para no perder la cordura esperando
    print(f"[{n_atomos}/100 átomos] MSE Exacto: {mse_exacto:.4f} | Esparcidad: {esparcidad_pct:.2f}% | Tiempo iteración: {tiempo_iteracion:.1f}s")

tiempo_total = (time.time() - tiempo_inicio_total) / 60
print("-" * 60)
print(f"¡Proceso completado en {tiempo_total:.2f} minutos!")

# -------------------------------------------------------------------
# 3. GRAFICAR LA CURVA COMPLETA DE COMPROMISO
# -------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

# Eje izquierdo: Error Exacto (MSE)
color1 = 'tab:red'
ax1.set_xlabel('Número de Átomos en el Diccionario (Complejidad)', fontsize=12)
ax1.set_ylabel('Error Cuadrático Medio Exacto (MSE)', color=color1, fontsize=12)
ax1.plot(rango_atomos, resultados_mse, color=color1, linewidth=2.5, label='MSE Exacto')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.5)

# Eje derecho: Esparcidad (Compresión)
ax2 = ax1.twinx()  
color2 = 'tab:blue'
ax2.set_ylabel('Esparcidad (% de Ceros Exactos)', color=color2, fontsize=12)
ax2.plot(rango_atomos, resultados_esparcidad, color=color2, linewidth=2.5, linestyle='dotted', label='Esparcidad')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Evolución Exhaustiva: Error vs Esparcidad (4 a 100 átomos)', fontsize=14)
fig.tight_layout()
plt.show()