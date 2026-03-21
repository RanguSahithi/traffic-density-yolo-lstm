# Convert to HSV (for color detection)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Detect road-like colors (gray/black)
lower_gray = np.array([0, 0, 50])
upper_gray = np.array([180, 50, 200])

mask = cv2.inRange(hsv, lower_gray, upper_gray)
road_pixels = np.sum(mask > 0)

st.image(frame, channels="BGR")

# 🚨 VALIDATION
if road_pixels < 5000:
    st.error("❌ Not a traffic image. Please upload a road image.")
else:
    # Then apply edge logic
    if edge_count < 5000:
        level = "Low Traffic 🟢"
    elif edge_count < 15000:
        level = "Medium Traffic 🟡"
    else:
        level = "High Traffic 🔴"

    st.success(f"Traffic Level: {level}")