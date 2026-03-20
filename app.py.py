import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

vehicle_classes = [2, 3, 5, 7]

st.title("🚦 Traffic Density Detection (YOLO)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    results = model(frame)

    count = sum(int(box.cls) in vehicle_classes for box in results[0].boxes)
    annotated = results[0].plot()

    st.image(annotated, channels="BGR")
    st.write(f"Vehicle Count: {count}")