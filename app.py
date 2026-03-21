import streamlit as st
import cv2
import numpy as np

st.title("🚦 Traffic Density Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    st.image(frame, channels="BGR")

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)
    edge_count = np.sum(edges > 0)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Road color detection
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 200])
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    road_pixels = np.sum(mask > 0)

    # 🚨 VALIDATION
    if road_pixels < 5000:
        st.error("❌ Not a traffic image. Please upload a road image.")
    else:
        if edge_count < 5000:
            level = "Low Traffic 🟢"
        elif edge_count < 15000:
            level = "Medium Traffic 🟡"
        else:
            level = "High Traffic 🔴"

        st.success(f"Traffic Level: {level}")