# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blur, 50, 150)

edge_count = np.sum(edges > 0)

st.image(frame, channels="BGR")

# 🚨 Check if it's even a traffic-like image
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