from deepface import DeepFace

result = DeepFace.verify(
    img1_path="reference/reference.jpeg",
    img2_path="test/test2.jpeg",
    model_name="ArcFace",
    detector_backend="retinaface",
    distance_metric="cosine",
    enforce_detection=True
)

print(result)
