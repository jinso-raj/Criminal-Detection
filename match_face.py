import numpy as np
from deepface import DeepFace
from numpy.linalg import norm

EMB_PATH = "embeddings/person1.npy"
TEST_IMAGE = "test/cctv_frame.jpg"
THRESHOLD = 0.4

embeddings = np.load(EMB_PATH)

test_rep = DeepFace.represent(
    img_path=TEST_IMAGE,
    model_name="ArcFace",
    detector_backend="retinaface",
    enforce_detection=True
)

test_emb = test_rep[0]["embedding"]

distances = []

for ref_emb in embeddings:
    cosine_sim = np.dot(ref_emb, test_emb) / (
        norm(ref_emb) * norm(test_emb)
    )
    distance = 1 - cosine_sim
    distances.append(distance)

min_distance = min(distances)

print("Best distance:", min_distance)

if min_distance < THRESHOLD:
    print(" MATCH FOUND")
else:
    print(" NO MATCH")
