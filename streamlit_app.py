import cv2
import numpy as np
import os
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Mask Detection", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "assets", "mask_detector.h5")
PROTOTXT_PATH = os.path.join(BASE_DIR, "assets", "face_detector", "deploy.prototxt")
WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "face_detector",
    "res10_300x300_ssd_iter_140000.caffemodel",
)

def verify_assets():
    missing = [p for p in [MODEL_PATH, PROTOTXT_PATH, WEIGHTS_PATH] if not os.path.exists(p)]
    if missing:
        st.error("Missing model files. Please redeploy with the assets folder included.")
        for path in missing:
            st.write(path)
        st.stop()


@st.cache_resource
def load_assets():
    verify_assets()
    model = load_model(MODEL_PATH)
    face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, WEIGHTS_PATH)
    return model, face_net

model, face_net = load_assets()

st.title("Face Mask Detection")
st.write("Use the camera or upload an image.")

camera_image = st.camera_input("Take a picture")
uploaded_image = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

image_bytes = None
if camera_image is not None:
    image_bytes = camera_image.getvalue()
elif uploaded_image is not None:
    image_bytes = uploaded_image.getvalue()

if image_bytes is None:
    st.stop()

image_array = np.frombuffer(image_bytes, dtype=np.uint8)
frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
if frame is None:
    st.error("Could not read the image.")
    st.stop()

(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
face_net.setInput(blob)
detections = face_net.forward()

faces = []
locs = []

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w - 1, endX), min(h - 1, endY)

        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((startX, startY, endX, endY))

preds = []
if len(faces) > 0:
    faces = np.array(faces, dtype="float32")
    preds = model.predict(faces, batch_size=32)

for (box, pred) in zip(locs, preds):
    (startX, startY, endX, endY) = box
    mask, without_mask = pred
    label = "Mask" if mask > without_mask else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    score = max(mask, without_mask)

    cv2.putText(
        frame,
        f"{label}: {score * 100:.2f}%",
        (startX, startY - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2,
    )
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
st.image(frame_rgb, channels="RGB")
