
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 PlantGuard AI")
st.subheader("Explainable Plant Disease Detection Using CNN")

st.write(
    "Upload a plant leaf image to identify its disease "
    "and view a Grad-CAM visual explanation."
)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

@st.cache_data
def load_classes():
    with open("class_names.json", "r") as f:
        return json.load(f)

model = load_model()
class_names = load_classes()

# Build Grad-CAM model
def build_grad_model(model):
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = inputs
    conv_output = None

    for layer in model.layers:
        x = layer(x, training=False)

        if layer.name == "last_conv":
            conv_output = x

    grad_model = tf.keras.Model(
        inputs=inputs,
        outputs=[conv_output, x]
    )

    return grad_model

grad_model = build_grad_model(model)

# Generate Grad-CAM
def generate_gradcam(image):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            image, training=False
        )
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    gradients = tape.gradient(loss, conv_outputs)

    if gradients is None:
        raise ValueError("Unable to calculate Grad-CAM gradients.")

    pooled_gradients = tf.reduce_mean(
        gradients, axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients, axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), int(predicted_class.numpy()), predictions[0].numpy()

# Image upload
uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    original_image = Image.open(uploaded_file).convert("RGB")

    st.image(
        original_image,
        caption="Uploaded Leaf Image",
        width=300
    )

    # Preprocess image
    resized_image = original_image.resize((128, 128))
    image_array = np.array(resized_image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    if st.button("Analyze Plant Image"):

        with st.spinner("Analyzing image..."):

            heatmap, predicted_class, probabilities = (
                generate_gradcam(image_array)
            )

            predicted_label = class_names[predicted_class]
            confidence = probabilities[predicted_class] * 100

            st.success("Analysis completed!")

            st.subheader("Prediction Result")
            st.write("**Predicted Class:**", predicted_label.replace("_", " "))
            st.write(f"**Confidence:** {confidence:.2f}%")

            # Display class probabilities
            st.subheader("Class Probabilities")

            for i, name in enumerate(class_names):
                st.write(
                    f"{name.replace('_', ' ')}: "
                    f"{probabilities[i] * 100:.2f}%"
                )

            # Generate heatmap overlay
            heatmap_resized = cv2.resize(
                heatmap, (original_image.width, original_image.height)
            )

            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap_resized),
                cv2.COLORMAP_JET
            )

            heatmap_colored = cv2.cvtColor(
                heatmap_colored, cv2.COLOR_BGR2RGB
            )

            original_array = np.array(original_image)

            overlay = cv2.addWeighted(
                original_array, 0.6,
                heatmap_colored, 0.4, 0
            )

            st.subheader("Grad-CAM Visual Explanation")

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

            st.info(
                "Grad-CAM highlights image regions that influenced "
                "the model's prediction. It is an explanation aid, "
                "not a definitive disease diagnosis."
            )
