
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import json

st.set_page_config(
    page_title="ASL Alphabet Recognition",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 ASL Alphabet Recognition")
st.write("Upload an image of an American Sign Language hand gesture.")

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        BASE_DIR / "best_cnn_model.keras"
    )

model = load_model()

with open(BASE_DIR / "class_names.json", "r") as f:
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
    image_array = np.array(processed_image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    st.success(f"Predicted Sign: {predicted_class}")
    st.info(f"Model confidence: {confidence:.2f}%")
    st.caption(
        "Confidence is the model's score, not a guarantee that "
        "the prediction is correct."
    )
