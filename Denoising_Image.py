import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn.datasets import fetch_olivetti_faces

# 1. Carga de datos
print("Cargando rostros de Olivetti...")
faces = fetch_olivetti_faces().images
train_data = faces[:40] 

# 2. Extracción de parches
patch_size = (8, 8)
print("Extrayendo parches...")
patches = extract_patches_2d(train_data, patch_size, max_patches=10000, random_state=0)
patches = patches.reshape(patches.shape[0], -1)

# Normalización: El descenso de gradiente funciona mejor si los datos están centrados
patches -= np.mean(patches, axis=0)
patches /= (np.std(patches, axis=0) + 1e-3)

# 3. MiniBatchDictionaryLearning (Corregido para scikit-learn moderno)
print("Entrenando Diccionario Disperso...")
start_time = time()
mbdl = MiniBatchDictionaryLearning(
    n_components=150, 
    batch_size=200, 
    alpha=1.0,      
    max_iter=50,     # <--- CAMBIO AQUÍ: 'n_iter' ahora es 'max_iter'
    random_state=42
)
dictionary = mbdl.fit(patches).components_
print(f"Entrenamiento completado en {time() - start_time:.2f}s")

# 4. Suceso ruidoso
test_face = faces[50]
noisy_face = test_face + 0.15 * np.random.randn(*test_face.shape)

# 5. Denoising
test_patches = extract_patches_2d(noisy_face, patch_size)
test_patches_flat = test_patches.reshape(test_patches.shape[0], -1)

# Codificación dispersa: resolvemos el problema de optimización para encontrar los alphas
sparse_codes = mbdl.transform(test_patches_flat)

# Reconstrucción combinando los átomos del diccionario
denoised_patches = np.dot(sparse_codes, dictionary)
denoised_patches = denoised_patches.reshape(-1, *patch_size)
denoised_face = reconstruct_from_patches_2d(denoised_patches, test_face.shape)

# 6. Resultados visuales
plt.figure(figsize=(12, 4))
plt.subplot(131); plt.imshow(test_face, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(132); plt.imshow(noisy_face, cmap='gray'); plt.title("Con Ruido"); plt.axis('off')
plt.subplot(133); plt.imshow(denoised_face, cmap='gray'); plt.title("Denoised"); plt.axis('off')
plt.show()