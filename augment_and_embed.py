import cv2
import os
import numpy as np
from deepface import DeepFace

INPUT_IMAGE = "reference/person1/person1.jpg"
AUG_DIR = "augmented/person1"
EMB_PATH = "embeddings/person1.npy"

os.makedirs(AUG_DIR, exist_ok=True)

img = cv2.imread(INPUT_IMAGE)

augmentations = []

# Original
augmentations.append(img)

# Brightness variations
for alpha in [0.7, 1.3]:
    aug = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    augmentations.append(aug)

# Blur (CCTV effect)
for k in [3, 5]:
    aug = cv2.GaussianBlur(img, (k, k), 0)
    augmentations.append(aug)

# Small rotations
h, w = img.shape[:2]
center = (w // 2, h // 2)
for angle in [-10, -5, 5, 10]:
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aug = cv2.warpAffine(img, M, (w, h))
    augmentations.append(aug)

# Save augmented images
image_paths = []
for i, aug in enumerate(augmentations):
    path = f"{AUG_DIR}/aug_{i}.jpg"
    cv2.imwrite(path, aug)
    image_paths.append(path)

print(f"Saved {len(image_paths)} augmented images.")


embeddings = []

for path in image_paths:
    try:
        rep = DeepFace.represent(
            img_path=path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )
        embeddings.append(rep[0]["embedding"])
    except:
        print("Face not detected in", path)

embeddings = np.array(embeddings)
np.save(EMB_PATH, embeddings)

print("Saved embeddings shape:", embeddings.shape)

