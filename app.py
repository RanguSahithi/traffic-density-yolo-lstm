import streamlit as st
import cv2
import numpy as np

st.title("🚦 Traffic Density Detection (Demo)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Fake density calculation (for demo)
    density = np.mean(gray)

    st.image(frame, channels="BGR")

    if density < 85:
        level = "Low Traffic 🟢"
    elif density < 170:
        level = "Medium Traffic 🟡"
    else:
        level = "High Traffic 🔴"

    st.write(f"Traffic Level: {level}")