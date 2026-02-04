import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------
# 1) LOAD IMAGE DATASET FROM SKLEARN
# ------------------------------------------------------

# Olivetti faces: 400 grayscale images of size 64x64
dataset = fetch_olivetti_faces()

images = dataset.images   # shape: (400, 64, 64)

print("Dataset shape:", images.shape)

# ------------------------------------------------------
# 2) EXTRACT PATCHES (16x16)
# ------------------------------------------------------

patch_size = (16, 16)

# Extract many random patches from all images
patches = []

for img in images:
    p = extract_patches_2d(img, patch_size, max_patches=50, random_state=0)
    patches.append(p)

patches = np.concatenate(patches, axis=0)

print("Number of patches:", patches.shape[0])

# Flatten each patch into a vector
patches = patches.reshape(patches.shape[0], -1)

# ------------------------------------------------------
# Normalize patches (important for dictionary learning)
# ------------------------------------------------------

scaler = StandardScaler(with_mean=True, with_std=True)
patches = scaler.fit_transform(patches)

# ------------------------------------------------------
# 3) LEARN SPARSE DICTIONARY
# ------------------------------------------------------

n_components = 100   # number of dictionary atoms (basis patches)

dict_learner = MiniBatchDictionaryLearning(
    n_components=n_components,
    alpha=1,              # sparsity regularization (higher = sparser)
    batch_size=256,
    random_state=0
)

# Learn dictionary
dictionary = dict_learner.fit(patches).components_

print("Dictionary shape:", dictionary.shape)
# (n_components, 256) since 16x16 = 256 pixels

# ------------------------------------------------------
# 4) COMPUTE SPARSE REPRESENTATIONS (CODES)
# ------------------------------------------------------

sparse_codes = dict_learner.transform(patches)

print("Sparse code shape:", sparse_codes.shape)

# Each patch ≈ sparse linear combination of dictionary atoms

# ------------------------------------------------------
# 5) VISUALIZE SOME LEARNED DICTIONARY ATOMS
# ------------------------------------------------------

fig, axes = plt.subplots(10, 10, figsize=(6, 6))

for i, ax in enumerate(axes.flat):
    atom = dictionary[i].reshape(16, 16)
    ax.imshow(atom, cmap="gray")
    ax.axis("off")

plt.suptitle("Learned dictionary patches")
plt.show()
