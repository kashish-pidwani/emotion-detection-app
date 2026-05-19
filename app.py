import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# ---------------- LABELS ----------------
class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# ---------------- EMOJIS ----------------
emotion_emojis = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐"
}

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>😊 Emotion Detection AI App</h1>",
    unsafe_allow_html=True
)

st.write("Upload an image and detect emotion instantly")

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Choose Image", type=["jpg", "jpeg", "png"])

# ---------------- PROCESS ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        st.write("Processing...")

        # ---------------- PREPROCESS (GRAYSCALE FIX) ----------------
        image = image.convert("L")   # IMPORTANT for your model
        image = image.resize((48, 48))

        img_array = np.array(image)
        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        # ---------------- PREDICT ----------------
        prediction = model.predict(img_array)

        pred_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        # safe label mapping
        if pred_index < len(class_names):
            result = class_names[pred_index]
        else:
            result = "Unknown"

        emoji = emotion_emojis.get(result, "")

        # ---------------- OUTPUT UI (FIXED VISIBILITY) ----------------
        st.markdown(
            f"""
            <div style='
                text-align:center;
                padding:30px;
                border-radius:15px;
                background-color:#111827;
                color:white;
                box-shadow:0px 4px 15px rgba(0,0,0,0.3);
            '>
                <h1 style='color:#4CAF50;'>{emoji} {result}</h1>
                <p style='font-size:18px;'>Confidence: {confidence:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Error during prediction: {e}")