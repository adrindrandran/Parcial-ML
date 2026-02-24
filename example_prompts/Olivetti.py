import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------
# 1) LOAD IMAGE DATASET
# ------------------------------------------------------

dataset = fetch_olivetti_faces()
images = dataset.images  # (400, 64, 64)

print("Dataset shape:", images.shape)

# ------------------------------------------------------
# 2) TRAIN / TEST SPLIT
# ------------------------------------------------------

train_images, test_images = train_test_split(
    images, test_size=0.2, random_state=0
)

print("Train images:", train_images.shape)
print("Test images:", test_images.shape)

# ------------------------------------------------------
# 3) EXTRACT PATCHES FROM TRAIN SET
# ------------------------------------------------------

patch_size = (16, 16)

patches = []

for img in train_images:
    p = extract_patches_2d(img, patch_size, max_patches=50, random_state=0)
    patches.append(p)

patches = np.concatenate(patches, axis=0)

print("Number of training patches:", patches.shape[0])

# Flatten patches
patches = patches.reshape(patches.shape[0], -1)

# Normalize
scaler = StandardScaler(with_mean=True, with_std=True)
patches = scaler.fit_transform(patches)

# ------------------------------------------------------
# 4) LEARN DICTIONARY (TRAINING)
# ------------------------------------------------------

n_components = 100

dict_learner = MiniBatchDictionaryLearning(
    n_components=n_components,
    alpha=1,
    batch_size=256,
    random_state=0
)

dictionary = dict_learner.fit(patches).components_

print("Dictionary shape:", dictionary.shape)

# ------------------------------------------------------
# 5) TEST: RECONSTRUCT ONE TEST IMAGE
# ------------------------------------------------------

test_image = test_images[0]

# Extract ALL patches from test image
test_patches = extract_patches_2d(test_image, patch_size)
n_test_patches = test_patches.shape[0]

# Flatten
test_patches_flat = test_patches.reshape(n_test_patches, -1)

# Normalize with TRAIN scaler
test_patches_flat = scaler.transform(test_patches_flat)

# Sparse coding
test_codes = dict_learner.transform(test_patches_flat)

# Reconstruct patches
reconstructed_patches = np.dot(test_codes, dictionary)

# Undo normalization
reconstructed_patches = scaler.inverse_transform(reconstructed_patches)

# Reshape to 16x16
reconstructed_patches = reconstructed_patches.reshape(
    n_test_patches, patch_size[0], patch_size[1]
)

# Reconstruct full image
reconstructed_image = reconstruct_from_patches_2d(
    reconstructed_patches, test_image.shape
)

# ------------------------------------------------------
# 6) COMPUTE RECONSTRUCTION ERROR
# ------------------------------------------------------

mse = mean_squared_error(test_image, reconstructed_image)
print("Reconstruction MSE (1 image):", mse)

# ------------------------------------------------------
# 7) VISUAL COMPARISON
# ------------------------------------------------------

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(test_image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(reconstructed_image, cmap="gray")
plt.title("Reconstructed Image")
plt.axis("off")

plt.show()

# ------------------------------------------------------
# 8) OPTIMIZADO: ERROR PROMEDIO CON MUESTREO ALEATORIO
# ------------------------------------------------------

errors = []
max_p = 100  # Reducimos de 2,401 a solo 100 parches por imagen

print(f"Calculando MSE promedio usando {max_p} parches aleatorios por imagen...")

for img in test_images:
    # Solo extraemos una muestra aleatoria (mucho más rápido)
    patches = extract_patches_2d(img, patch_size, max_patches=max_p, random_state=0)
    
    # El resto del proceso es igual, pero con 24 veces menos datos
    n_p = patches.shape[0]
    patches_flat = patches.reshape(n_p, -1)
    patches_flat = scaler.transform(patches_flat)

    codes = dict_learner.transform(patches_flat)
    rec_patches_flat = np.dot(codes, dictionary)
    rec_patches_flat = scaler.inverse_transform(rec_patches_flat)

    # Para calcular el MSE de parches individuales vs originales
    # No necesitamos reconstruir la imagen completa (reconstruct_from_patches_2d)
    # porque ya tenemos los parches originales para comparar.
    err = mean_squared_error(patches.reshape(n_p, -1), rec_patches_flat)
    errors.append(err)

print("Average test MSE (estimated):", np.mean(errors))
# ------------------------------------------------------
# 9) VISUALIZACIÓN DE ÁTOMOS Y COEFICIENTES (AÑADIDO)
# ------------------------------------------------------

# A. Visualizar los primeros 16 "Átomos" del Diccionario
plt.figure(figsize=(6, 6))
plt.suptitle("Diccionario: Los 16 primeros 'Átomos' (Formas básicas)", fontsize=14)
for i in range(16):
    plt.subplot(4, 4, i + 1)
    # Recordamos que cada átomo es un vector de 256, lo volvemos a hacer 16x16
    plt.imshow(dictionary[i].reshape(patch_size), cmap="gray")
    plt.axis("off")
plt.show()

# B. Visualizar la Matriz de Coeficientes de la reconstrucción anterior
# Tomamos los coeficientes de los primeros 50 parches de la imagen de test
plt.figure(figsize=(10, 4))
plt.imshow(test_codes[:50].T, aspect='auto', cmap='viridis')
plt.colorbar(label="Peso del átomo")
plt.title("Matriz de Coeficientes (Sparse Codes)")
plt.xlabel("Índice del Parche (en la imagen)")
plt.ylabel("Índice del Átomo (del 0 al 99)")
plt.tight_layout()
plt.show()

print("\nInterpretación:")
print("- Los Átomos son las formas 'maestras' aprendidas.")
print("- La Matriz de Coeficientes muestra qué átomos se activan para cada parte de la cara.")
print("- Las zonas oscuras en la matriz significan 'peso cero' (Sparse), lo que hace eficiente al modelo.")