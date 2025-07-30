import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import base64
from io import BytesIO
from ultralytics import YOLO  

st.set_page_config(
    page_title="Crack Detection - Final Project",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.image("D:/my code/Crack_Detection/NTI.png", width=250)

def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_image_base64("image.png")  

st.markdown(f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{img_base64}" width="550">
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 5px;
    }
    .team-name {
        font-size: 22px;
        font-weight: bold;
        color: #FCAE56FF;
        text-align: center;
        margin-bottom: 30px;
    }
    .footer {
        font-size: 13px;
        color: #aaa;
        text-align: center;
        margin-top: 50px;
    }
    </style>
    <div class='main-title'> Crack Detection System</div>
    <div class='team-name'>🚀 SmartCrack Team: Beshoy Osama | Asmaa Elkashif | Omnia Eldeeb | Omar Mohamed </div>
""", unsafe_allow_html=True)

model = YOLO("best (1).pt")


def center_image(img, width=500):
    buffered = BytesIO()
    img_pil = Image.fromarray(img)
    img_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    html_code = f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_base64}" width="{width}px">
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    with st.spinner("🔍 Detecting cracks..."):
        try:
            st.info("🚀 Starting prediction...")
            results = model.predict(image, conf=0.3)

            if len(results[0].boxes) == 0:
                st.warning("😢 No cracks detected in the image.")
            else:
                st.success("✅ Cracks detected successfully!")
                result_img = results[0].plot()
                st.subheader("Detection Result (Crack)")
                center_image(result_img, width=500)
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")


st.markdown("""
    <div class='footer'>
        Graduation Project - Computer Vision Internship - 2025
    </div>
""", unsafe_allow_html=True)

