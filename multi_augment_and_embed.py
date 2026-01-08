import cv2
import os
import numpy as np
from deepface import DeepFace

REF_DIR = "reference/person1"
AUG_DIR = "augmented/person1"
EMB_PATH = "embeddings/person1.npy"

os.makedirs(AUG_DIR, exist_ok=True)

augmentations = []

def augment_image(img):
    aug_list = []
    aug_list.append(img)

    # Brightness
    for alpha in [0.7, 1.3]:
        aug_list.append(cv2.convertScaleAbs(img, alpha=alpha, beta=0))

    # Blur
    for k in [3, 5]:
        aug_list.append(cv2.GaussianBlur(img, (k, k), 0))

    # Rotation
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-10, -5, 5, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_list.append(cv2.warpAffine(img, M, (w, h)))

    return aug_list


image_paths = []

# 🔥 LOOP OVER MULTIPLE REAL IMAGES
for file in os.listdir(REF_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(REF_DIR, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        aug_imgs = augment_image(img)

        for aug in aug_imgs:
            fname = f"aug_{len(image_paths)}.jpg"
            path = os.path.join(AUG_DIR, fname)
            cv2.imwrite(path, aug)
            image_paths.append(path)

print(f"Total augmented images: {len(image_paths)}")


# 🧠 Extract embeddings
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
 