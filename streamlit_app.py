import cv2
import numpy as np
import os
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
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
st.write("Webcam mask detection.")
st.caption("Use fallback capture for stability, or enable live mode (beta).")

if "live_mode" not in st.session_state:
    st.session_state.live_mode = False

if not st.session_state.live_mode:
    if st.button("Enable live stream (beta)"):
        st.session_state.live_mode = True
        st.rerun()


def annotate_frame(frame_bgr, model_obj, face_net_obj):
    (h, w) = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net_obj.setInput(blob)
    detections = face_net_obj.forward()

    faces = []
    locs = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face = frame_bgr[startY:endY, startX:endX]
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
        preds = model_obj.predict(faces, batch_size=32)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        mask, without_mask = pred
        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        score = max(mask, without_mask)

        cv2.putText(
            frame_bgr,
            f"{label}: {score * 100:.2f}%",
            (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )
        cv2.rectangle(frame_bgr, (startX, startY), (endX, endY), color, 2)

    return frame_bgr


class MaskVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.face_net = face_net

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_bgr = annotate_frame(frame_bgr, self.model, self.face_net)

        return frame.from_ndarray(frame_bgr, format="bgr24")


if st.session_state.live_mode:
    st.info("Live mode is enabled. If it fails on your network, refresh and use fallback capture.")
    webrtc_streamer(
        key="mask_detect_live",
        video_processor_factory=MaskVideoProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        },
        desired_playing_state=False,
        async_processing=True,
    )
else:
    captured = st.camera_input("Fallback: Take photo and run mask detection")
    if captured is not None:
        file_bytes = np.asarray(bytearray(captured.read()), dtype=np.uint8)
        snap = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if snap is not None:
            annotated = annotate_frame(snap, model, face_net)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
