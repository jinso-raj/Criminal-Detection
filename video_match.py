import cv2
import numpy as np
from deepface import DeepFace
from numpy.linalg import norm
from collections import deque

# ---------------- CONFIG ----------------
VIDEO_PATH = "video/yes.mp4"  # or 0 for webcam
EMB_PATH = "embeddings/person1.npy"
THRESHOLD = 0.4

CONFIRM_FRAMES = 3       # how many matches
WINDOW_SIZE = 15         # frames window
# ----------------------------------------

embeddings = np.load(EMB_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

match_window = deque(maxlen=WINDOW_SIZE)
alert_triggered = False 

def min_distance(test_emb, ref_embeddings):
    distances = []
    for ref in ref_embeddings:
        cosine_sim = np.dot(ref, test_emb) / (
            norm(ref) * norm(test_emb)
        )
        distances.append(1 - cosine_sim)
    return min(distances)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="retinaface",
            enforce_detection=False
        )

        for face in faces:
            face_img = face["face"]

            rep = DeepFace.represent(
                img_path=face_img,
                model_name="ArcFace",
                detector_backend="skip"
            )

            test_emb = rep[0]["embedding"]
            distance = min_distance(test_emb, embeddings)

            is_match = distance < THRESHOLD
            match_window.append(is_match)

            # Draw bounding box
            x, y, w, h = face["facial_area"].values()
            color = (0, 255, 0) if is_match else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            cv2.putText(
                frame,
                f"dist: {distance:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            # Confirmation logic
            if match_window.count(True) >= CONFIRM_FRAMES and not alert_triggered:
                alert_triggered = True
                print("🚨 CONFIRMED MATCH DETECTED")

    except Exception as e:
        pass

    cv2.imshow("Criminal Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
