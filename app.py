import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load models
yolo_model = YOLO("yolov8n.pt")
lstm_model = load_model("lstm_model.h5")

vehicle_classes = [2, 3, 5, 7]

scaler = MinMaxScaler()

st.title("🚦 Traffic Density + Prediction")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    results = yolo_model(frame)

    count = sum(int(box.cls) in vehicle_classes for box in results[0].boxes)
    annotated = results[0].plot()

    st.image(annotated, channels="BGR")
    st.write(f"Vehicle Count: {count}")

    # ---- LSTM Prediction (simple demo input) ----
    dummy_data = np.array([[count]*10])  # fake sequence
    dummy_data = dummy_data.reshape(1,10,1)

    prediction = lstm_model.predict(dummy_data, verbose=0)
    predicted_value = int(prediction[0][0])

    st.write(f"🔮 Predicted Traffic (next minute): {predicted_value}")