import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# Vehicle class IDs (COCO dataset)
vehicle_classes = [2, 3, 5, 7]  # car, bike, bus, truck

st.title("🚦 Traffic Density Estimation")

st.info("LSTM prediction module is implemented separately due to deployment limitations.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # YOLO Detection
    results = yolo_model(frame)

    # Count vehicles
    count = sum(int(box.cls) in vehicle_classes for box in results[0].boxes)

    # Annotated image
    annotated = results[0].plot()

    st.image(annotated, channels="BGR")
    st.write(f"🚗 Vehicle Count: {count}")