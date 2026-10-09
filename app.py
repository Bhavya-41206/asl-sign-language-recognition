
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(
    page_title="ASL Alphabet Recognition",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 ASL Alphabet Recognition")
st.write("Upload an image of an American Sign Language hand gesture.")

model = tf.keras.models.load_model(
    "/kaggle/working/best_cnn_model.keras"
)

with open("/kaggle/working/class_names.json", "r") as f:
    class_names = json.load(f)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    processed_image = image.resize((128, 128))
    image_array = np.array(processed_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    st.success(f"Predicted Sign: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
