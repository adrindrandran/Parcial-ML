import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn.datasets import fetch_olivetti_faces

# 1. Carga de datos
print("Cargando rostros de Olivetti...")
faces = fetch_olivetti_faces().images # Cada cara es 64x64

# 2. Extracción de parches REFORMULADA
patch_size = (8, 8)
print("Extrayendo parches de forma manual para asegurar dimensiones...")

all_patches = []
for i in range(50): # Usamos 50 caras para entrenar
    # Extraemos parches de CADA cara individualmente
    p = extract_patches_2d(faces[i], patch_size, max_patches=100, random_state=i)
    all_patches.append(p)

# Convertimos a un solo array de (N_PARCHES, 8, 8) -> (5000, 8, 8)
patches = np.vstack(all_patches)
# Aplanamos a (5000, 64)
patches = patches.reshape(patches.shape[0], -1)

# Normalización estándar para ML (Media 0, Varianza 1)
patches -= np.mean(patches, axis=0)
patches /= (np.std(patches, axis=0) + 1e-3)

# 3. Entrenamiento (Ahora sí, sobre 64 features)
print(f"Entrenando Diccionario sobre {patches.shape[0]} parches de dimensión {patches.shape[1]}...")
start_time = time()
mbdl = MiniBatchDictionaryLearning(
    n_components=121, # Un número cuadrado para visualizarlo bien luego (11x11)
    batch_size=100, 
    alpha=1.0,      
    max_iter=100,
    random_state=42
)
mbdl.fit(patches)
dictionary = mbdl.components_
print(f"Entrenamiento completado en {time() - start_time:.2f}s")

# 4. El "Suceso" Ruidoso
test_face = faces[60] # Una cara que no usamos para entrenar
noisy_face = test_face + 0.1 * np.random.randn(*test_face.shape)

# 5. Denoising
# Extraer parches de la imagen con ruido
test_patches = extract_patches_2d(noisy_face, patch_size)
test_patches_flat = test_patches.reshape(test_patches.shape[0], -1)

# IMPORTANTE: El transform debe recibir la misma dimensión que el fit (64)
sparse_codes = mbdl.transform(test_patches_flat)

# Reconstrucción: sumamos los átomos + volvemos a dar forma de imagen
denoised_patches = np.dot(sparse_codes, dictionary)
denoised_patches = denoised_patches.reshape(-1, *patch_size)
denoised_face = reconstruct_from_patches_2d(denoised_patches, test_face.shape)

# 6. Visualización de la "Mente" de la IA y el resultado
plt.figure(figsize=(12, 5))

# Rostro Original vs Ruidoso vs Limpio
plt.subplot(1, 3, 1); plt.imshow(test_face, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(noisy_face, cmap='gray'); plt.title("Suceso Ruidoso"); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(denoised_face, cmap='gray'); plt.title("Denoised (Sparse)"); plt.axis('off')

plt.tight_layout()
plt.show()

# Extra: ¿Qué aprendió el diccionario?
plt.figure(figsize=(6, 6))
for i, atom in enumerate(dictionary[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(atom.reshape(8, 8), cmap='gray')
    plt.axis('off')
plt.suptitle("Los 100 'Genes Visuales' aprendidos", fontsize=14)
plt.show()