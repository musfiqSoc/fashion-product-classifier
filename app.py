import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# -----------------------------
# Load Model
# -----------------------------
model = load_model("fashion_mnist_cnn.keras")

# -----------------------------
# Class Names
# -----------------------------
class_names = [
    "T-shirt / Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

# -----------------------------
# App UI
# -----------------------------
st.title("👕 Fashion Product Classifier")

st.write(
    "Upload a fashion image and our CNN model "
    "will classify it into one of 10 Fashion-MNIST categories."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # Display original image
    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    # Preprocessing
    image = image.convert("L")
    image = image.resize((28, 28))

    image_array = np.array(image)
    image_array = image_array.astype("float32") / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(image_array, verbose=0)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Result
    st.subheader("Prediction")

    st.success(
        f"{class_names[predicted_class]}"
    )

    st.write(
        f"Confidence: **{confidence * 100:.2f}%**"
    )