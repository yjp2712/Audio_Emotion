import os
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# Streamlit Configuration
# ============================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

TARGET_SIZE = (224, 224)

# ============================================================
# Custom Layers
# ============================================================

class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        patch_dims = tf.shape(patches)[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])

        return patches


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()

        self.projection = tf.keras.layers.Dense(projection_dim)
        self.num_patches = num_patches
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_patches,
            output_dim=self.projection_dim
        )

    def call(self, patches):
        positions = tf.range(
            start=0,
            limit=self.num_patches,
            delta=1
        )

        return self.projection(patches) + self.position_embedding(positions)


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_model(language):

    folder = f"{language}_Saved_Model"

    model = tf.keras.models.load_model(
        os.path.join(folder, "Hybrid_VGG16_ViT.keras"),
        custom_objects={
            "PatchExtractor": PatchExtractor,
            "PatchEncoder": PatchEncoder
        }
    )

    with open(os.path.join(folder, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder


# ============================================================
# Spectrogram Generator
# ============================================================

def create_spectrogram(y, sr):

    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(np.abs(X))

    fig = plt.figure(figsize=(4, 4), dpi=64)

    plt.axis("off")

    librosa.display.specshow(
        Xdb,
        sr=sr,
        x_axis=None,
        y_axis=None,
        cmap="magma"
    )

    fig.canvas.draw()

    img = np.frombuffer(
        fig.canvas.tostring_rgb(),
        dtype=np.uint8
    )

    img = img.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )

    plt.close(fig)

    img = Image.fromarray(img)

    img = img.resize(TARGET_SIZE)

    return np.array(img)


# ============================================================
# Prediction Function
# ============================================================

def predict_emotion(audio_file, model, label_encoder):

    y, sr = librosa.load(audio_file, sr=22050)

    img = create_spectrogram(y, sr)

    x = img.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x, verbose=0)[0]

    idx = np.argmax(prediction)

    emotion = label_encoder.inverse_transform([idx])[0]

    confidence = prediction[idx]

    return emotion, confidence, prediction, img


# ============================================================
# UI
# ============================================================

st.title("🎤 Design and Implementation of Specific Novel Architecture for Bilingual Speech Emotion Recognization")
st.markdown("---")

language = st.selectbox(
    "Select Language",
    ["English", "Gujarati"]
)

model, label_encoder = load_model(language)

uploaded_file = st.file_uploader(
    "Upload WAV Audio",
    type=["wav"]
)

if uploaded_file is not None:

    st.markdown("## 🎵 Listen Audio")

    audio_bytes = uploaded_file.read()

    st.audio(audio_bytes, format="audio/wav")

    uploaded_file.seek(0)

    y, sr = librosa.load(uploaded_file, sr=22050)

    spectrogram = create_spectrogram(y, sr)

    st.markdown("## 📊 Generated Spectrogram")

    st.image(
        spectrogram,
        use_container_width=True
    )

    uploaded_file.seek(0)

    if st.button("🔍 Predict Emotion", use_container_width=True):

        with st.spinner("Recognizing emotion..."):

            emotion, confidence, probabilities, _ = predict_emotion(
                uploaded_file,
                model,
                label_encoder
            )

        st.success(f"### Predicted Emotion : {emotion}")

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.markdown("## 📈 Prediction Probabilities")

        classes = label_encoder.classes_

        for cls, prob in zip(classes, probabilities):

            st.write(f"**{cls}**")

            st.progress(float(prob))

            st.write(f"{prob*100:.2f}%")
