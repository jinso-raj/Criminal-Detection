import numpy as np
from deepface import DeepFace
from numpy.linalg import norm

#Extract embedding from reference image
ref_embedding = DeepFace.represent(
    img_path="reference/person2.jpeg",
    model_name="ArcFace",
    detector_backend="retinaface",
    enforce_detection=True
)

# ref_embedding is a list (in case multiple faces are found)
embedding_vector = ref_embedding[0]["embedding"]

# Save embedding
np.save("embeddings/person2.npy", embedding_vector)

print("Reference embedding saved.")
print("Embedding length:", len(embedding_vector))


# Load reference embedding
ref_embedding = np.load("embeddings/person2.npy")

# Extract embedding from test image
test_result = DeepFace.represent(
    img_path="test/test2.jpeg",
    model_name="ArcFace",
    detector_backend="retinaface",
    enforce_detection=True
)

test_embedding = test_result[0]["embedding"]

# Cosine similarity
cosine_similarity = np.dot(ref_embedding, test_embedding) / (
    norm(ref_embedding) * norm(test_embedding)
)

distance = 1 - cosine_similarity

print("Cosine similarity:", cosine_similarity)
print("Distance:", distance)

# Decision
THRESHOLD = 0.5

if distance < THRESHOLD:
    print("SAME PERSON")
else:
    print("DIFFERENT PERSON")