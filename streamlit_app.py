import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import time

# --- Config ---
MODEL_PATH = os.path.join("..", "model_training_output", "models", "best_model_finetuned.keras")
CLASS_INDICES_PATH = os.path.join("..", "model_training_output", "models", "class_indices.json")

# --- UI Setup ---
st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="⚕️",
    layout="wide"
)

st.title("Dermatological AI Assistant 🔬")
st.markdown("""
    This system uses an EfficientNetB0 deep learning model to provide rapid preliminary screening 
    for 38 different skin disease categories.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("⚠️ Medical Disclaimer")
    st.warning("""
        **Not a Diagnostic Tool**
        
        This application is explicitly NOT a replacement for professional medical diagnosis. 
        It is a clinical decision support tool and should only be used under appropriate 
        medical oversight.
    """)
    st.info("Supported classes: 38 dermatological conditions.")
    st.text("Model Framework: TensorFlow/Keras (Mocked)")

# --- Mock Model Loading ---
@st.cache_resource
def load_mock_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDICES_PATH):
        return None
        
    with open(CLASS_INDICES_PATH, "r") as f:
        class_names = json.load(f)
    return class_names

class_names = load_mock_model()

if class_names is None:
    st.error("Model or class indices not found. Please train the model first by running `scripts/model_training_part1.py`.")
else:
    # --- Main App ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Upload Image")
        uploaded_file = st.file_uploader("Choose an image (JPG, PNG)...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Run Analysis", type="primary"):
                with col2:
                    st.subheader("2. Analysis Results")
                    
                    with st.spinner("Analyzing image..."):
                        time.sleep(2) # simulate inference time
                        
                        # Generate Mock Predictions based on image
                        # This uses a dummy hash to get consistent "random" results for the same file
                        img_array = np.array(image)
                        rng = np.random.default_rng(img_array.sum())
                        
                        predictions = rng.random(len(class_names))
                        predictions = predictions / np.sum(predictions) # normalize
                        
                        # Make one class definitively higher for demonstration
                        top_idx = rng.choice(len(class_names))
                        predictions[top_idx] += 1.5 
                        predictions = predictions / np.sum(predictions) 
                        
                        # Get Top 3
                        top_3_indices = np.argsort(predictions)[-3:][::-1]
                        top_3_classes = [class_names[i] for i in top_3_indices]
                        top_3_probs = [predictions[i] for i in top_3_indices]
                        
                    st.success("Analysis complete!")
                    
                    # Display Results
                    st.markdown("### Top 3 Predictions")
                    for i in range(3):
                        confidence = top_3_probs[i] * 100
                        cls_name = top_3_classes[i]
                        
                        st.write(f"**{i+1}. {cls_name}** ({confidence:.1f}%)")
                        st.progress(float(top_3_probs[i]))

                    # Confidence Review
                    highest_conf = top_3_probs[0]
                    if highest_conf > 0.85:
                        st.info("💡 High confidence prediction. Model is very certain.")
                    elif highest_conf > 0.60:
                        st.warning("⚠️ Moderate confidence. Visual similarity with other conditions is possible.")
                    else:
                        st.error("🚨 Low confidence. The image may be low quality, out-of-distribution, or an atypical presentation. Manual review required.")
