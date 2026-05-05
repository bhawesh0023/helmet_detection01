import os

from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

# 🔥 Download model from Google Drive
file_id = "1KL67VORf-RKhuHyyDoTK4qNKLhQZKyYA"
url = f"https://drive.google.com/uc?id={file_id}"


# 🔥 Load model (only once)
model = YOLO("helmet1.pt")

# Page config
st.set_page_config(page_title="Helmet Detection", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>🪖 Smart Helmet Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image to detect helmet usage</p>", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📥 Uploaded Image", width=400)

    img_array = np.array(image)
    results = model(img_array, conf=0.5)

    # Detection logic
    helmet_detected = False
    head_detected = False

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if conf > 0.5:
                if label == "helmet":
                    helmet_detected = True
                if label in ["head", "person"]:
                    head_detected = True

    # Result text
    st.markdown("---")

    if helmet_detected:
        st.markdown("<h2 style='text-align: center; color: green;'>✅ Helmet Detected</h2>", unsafe_allow_html=True)
    elif head_detected:
        st.markdown("<h2 style='text-align: center; color: red;'>⚠️ No Helmet Detected</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center; color: orange;'>🤔 Not Sure</h2>", unsafe_allow_html=True)

    st.markdown("---")

    # Output image
    result_img = results[0].plot()
    st.image(result_img, caption="📤 Detection Result", width=450)
