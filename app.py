
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# 1. PROJECT FOLDER PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 PlantGuard AI")

st.subheader(
    "Explainable Plant Disease Detection Using CNN"
)

st.write(
    "Upload a plant leaf image to identify its disease "
    "and view a Grad-CAM visual explanation."
)


# ==========================================
# 3. LOAD TRAINED CNN MODEL
# ==========================================

@st.cache_resource
def load_model():

    model_path = BASE_DIR / "plant_disease_model.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            f"CNN model file not found: {model_path}"
        )

    return tf.keras.models.load_model(
        str(model_path)
    )


# ==========================================
# 4. LOAD CLASS NAMES
# ==========================================

@st.cache_data
def load_classes():

    class_path = BASE_DIR / "class_names.json"

    if not class_path.exists():
        raise FileNotFoundError(
            f"Class names file not found: {class_path}"
        )

    with open(class_path, "r") as file:
        classes = json.load(file)

    return classes


model = load_model()
class_names = load_classes()


# ==========================================
# 5. BUILD GRAD-CAM MODEL
# ==========================================

def build_grad_model(model):

    inputs = tf.keras.Input(
        shape=(128, 128, 3)
    )

    x = inputs
    conv_output = None

    for layer in model.layers:

        x = layer(
            x,
            training=False
        )

        if layer.name == "last_conv":
            conv_output = x

    if conv_output is None:
        raise ValueError(
            "The last_conv layer was not found."
        )

    grad_model = tf.keras.Model(
        inputs=inputs,
        outputs=[
            conv_output,
            x
        ]
    )

    return grad_model


grad_model = build_grad_model(model)


# ==========================================
# 6. GENERATE GRAD-CAM
# ==========================================

def generate_gradcam(image):

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image,
            training=False
        )

        predicted_class = tf.argmax(
            predictions[0]
        )

        loss = predictions[:, predicted_class]

    gradients = tape.gradient(
        loss,
        conv_outputs
    )

    if gradients is None:
        raise ValueError(
            "Unable to calculate Grad-CAM gradients."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return (
        heatmap.numpy(),
        int(predicted_class.numpy()),
        predictions[0].numpy()
    )


# ==========================================
# 7. IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        original_image,
        caption="Uploaded Leaf Image",
        width=300
    )


    # ======================================
    # 8. IMAGE PREPROCESSING
    # ======================================

    resized_image = original_image.resize(
        (128, 128)
    )

    image_array = np.array(
        resized_image
    ).astype("float32") / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # ======================================
    # 9. ANALYZE PLANT IMAGE
    # ======================================

    if st.button("Analyze Plant Image"):

        with st.spinner("Analyzing image..."):

            try:

                heatmap, predicted_class, probabilities = (
                    generate_gradcam(image_array)
                )

                predicted_label = class_names[
                    predicted_class
                ]

                confidence = (
                    probabilities[predicted_class] * 100
                )

                st.success(
                    "Analysis completed successfully!"
                )


                # ==================================
                # 10. PREDICTION RESULT
                # ==================================

                st.subheader("Prediction Result")

                st.write(
                    "**Predicted Class:**",
                    predicted_label.replace("_", " ")
                )

                st.write(
                    f"**Confidence:** {confidence:.2f}%"
                )


                # ==================================
                # 11. CLASS PROBABILITIES
                # ==================================

                st.subheader("Class Probabilities")

                for i, name in enumerate(class_names):

                    st.write(
                        f"{name.replace('_', ' ')}: "
                        f"{probabilities[i] * 100:.2f}%"
                    )


                # ==================================
                # 12. GENERATE GRAD-CAM HEATMAP
                # ==================================

                heatmap_image = Image.fromarray(
                    np.uint8(255 * heatmap)
                )

                heatmap_resized = heatmap_image.resize(
                    (
                        original_image.width,
                        original_image.height
                    )
                )

                heatmap_resized = (
                    np.array(heatmap_resized) / 255.0
                )


                # Apply Matplotlib color map

                heatmap_colored = plt.get_cmap("jet")(
                    heatmap_resized
                )[..., :3]

                heatmap_colored = np.uint8(
                    255 * heatmap_colored
                )


                # ==================================
                # 13. CREATE GRAD-CAM OVERLAY
                # ==================================

                original_array = np.array(
                    original_image
                )

                overlay = np.uint8(
                    0.6 * original_array +
                    0.4 * heatmap_colored
                )


                # ==================================
                # 14. DISPLAY VISUAL EXPLANATION
                # ==================================

                st.subheader(
                    "Grad-CAM Visual Explanation"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        heatmap_colored,
                        caption="Grad-CAM Heatmap",
                        use_container_width=True
                    )

                with col2:

                    st.image(
                        overlay,
                        caption="Grad-CAM Overlay",
                        use_container_width=True
                    )


                # ==================================
                # 15. EXPLANATION
                # ==================================

                st.info(
                    "Grad-CAM highlights image regions that "
                    "influenced the CNN model's prediction. "
                    "It provides a visual explanation and "
                    "is not a definitive disease diagnosis."
                )

            except Exception as e:

                st.error(
                    f"An error occurred: {str(e)}"
                )
