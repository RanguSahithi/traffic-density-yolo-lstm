import streamlit as st
import cv2
import numpy as np

st.title("🚦 Traffic Density Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    edge_count = np.sum(edges > 0)

    st.image(frame, channels="BGR")

    # Validate image
    if edge_count < 1000:
        st.error("❌ Not a traffic image. Please upload a road/vehicle image.")
    else:
        if edge_count < 5000:
            level = "Low Traffic 🟢"
        elif edge_count < 15000:
            level = "Medium Traffic 🟡"
        else:
            level = "High Traffic 🔴"

        st.success(f"Traffic Level: {level}")