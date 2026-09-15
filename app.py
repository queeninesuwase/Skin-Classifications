import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="Skin Lesion AI Triage Portal", layout="wide")
st.title("Skin Lesion Multi-Class Diagnostic Portal")
st.markdown("### Machine Learning - Medical Imaging")

with st.sidebar:
    st.header("System Specifications")
    st.markdown("**Anatomical Target:** Human Skin")
    st.markdown("**Imaging Modality:** Dermoscopic Photos")
    st.markdown("**Core Backbone:** Pretrained MobileNetV2")
    st.error("CRITICAL DISCLAIMER: This platform is a student coursework prototype and NOT an approved medical diagnostic tool.")

MODEL_PATH = "skin_lesion_model.h5"

@st.cache_resource
def load_triage_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_triage_model()
    st.success("Neural Network weights loaded and cached successfully.")
except Exception as e:
    st.error(f"Failure loading model: {e}")
    st.stop()

def preprocess_incoming_scan(pil_image):
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    resized_img = pil_image.resize((224, 224))
    img_array = np.array(resized_img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

CLASS_LABELS = {
    0: "Actinic keratoses (akiec) [Pre-cancerous]",
    1: "Basal cell carcinoma (bcc) [Malignant]",
    2: "Benign keratosis-like lesions (bkl) [Benign]",
    3: "Dermatofibroma (df) [Benign]",
    4: "Melanoma (mel) [Highly Malignant]",
    5: "Melanocytic nevi (nv) [Benign Mole]",
    6: "Vascular lesions (vasc) [Benign]"
}

uploaded_file = st.file_uploader("Choose a dermoscopic jpeg/png image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    source_image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(source_image, use_container_width=True, caption="Source Scan Image File")
    with col2:
        with st.spinner("Running model forward pass..."):
            input_tensor = preprocess_incoming_scan(source_image)
            
            predictions = model.predict(input_tensor)
            
            pred_probabilities = predictions.flatten()
            
            predicted_idx = int(np.argmax(pred_probabilities))
            confidence = float(pred_probabilities[predicted_idx])
            final_class_text = CLASS_LABELS[predicted_idx]
           
            if confidence < 0.50:
                st.warning(
                    " **LOW CONFIDENCE DETECTED:** The model is highly uncertain about this specific skin tissue structure. "
                    "This image may be out-of-distribution or corrupted. Do not interpret this result without close physical inspection.",
                    icon="❗"
                )
            
            st.metric(label="Predicted Diagnosis Category", value=final_class_text)
            st.metric(label="Confidence Profile", value=f"{confidence * 100:.2f}%")
            st.progress(confidence)
            
            st.markdown("#### Plain-Language Clinical Translation")
            if "Malignant" in final_class_text or "Pre-cancerous" in final_class_text:
                st.markdown(f"**Summary:** The system flagged patterns matching **{final_class_text}** with **{confidence * 100:.1f}%** confidence. Immediate professional evaluation or biopsy is strongly recommended.")
            else:
                st.markdown(f"**Summary:** The model predicts **{final_class_text}** with **{confidence * 100:.1f}%** confidence. This is structurally aligned with a benign profile.")

st.markdown("---")
with st.expander("Local Application Setup Instructions (README Manual)"):
    st.markdown("""
    ### Local Environment Launch Manual
    Follow these structural commands within your local system terminal workspace environment:
    
    1. **Navigate directly into your assignment directory:**
       ```bash
       cd path/to/your/project-workspace
       ```
    2. **Activate your pre-existing coursework environment:**
       ```bash
       source Medical...venv/bin/activate
       ```
    3. **Boot up the local web-app framework instance:**
       ```bash
       streamlit run app.py
       ```
    """)
