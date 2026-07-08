import os
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import streamlit as st

import os
import json
import streamlit as st

# Read credentials from Streamlit Secrets
username = "yp271289"
api_key = "ffe074510c04ca2a400d94765a2e18f2"

# Create ~/.kaggle
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

# Write kaggle.json
with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
    json.dump(
        {
            "username": username,
            "key": api_key
        },
        f
    )

os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

# Download kernel output (only if not already downloaded)
if not os.path.exists("English_Saved_Model"):
    status = os.system("kaggle kernels output yp271289/own-data-model-save")

    if status != 0:
        st.error("Failed to download model from Kaggle.")
        st.stop()

    st.success("Model downloaded successfully.")
else:
    st.success("Model already exists.")

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

@tf.keras.utils.register_keras_serializable()
class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
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

        return tf.reshape(
            patches,
            [batch_size, -1, patch_dims]
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size
        })
        return config


@tf.keras.utils.register_keras_serializable()
class PatchEncoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_patches,
                 projection_dim,
                 **kwargs):

        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.projection_dim = projection_dim

        self.projection = tf.keras.layers.Dense(projection_dim)
        self.position_embedding = None

    def build(self, input_shape):
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_patches,
            output_dim=self.projection_dim,
        )

    def call(self, patches):
        positions = tf.range(0, self.num_patches)

        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_model(language):

    folder = f"{language}_Saved_Model"

    model_file = os.path.join(folder, "Hybrid_VGG16_ViT.keras")

    model = tf.keras.models.load_model(
        model_file,
        custom_objects={
            "PatchExtractor": PatchExtractor,
            "PatchEncoder": PatchEncoder,
        },
        compile=False,
    )

    with open(os.path.join(folder, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder


# ============================================================
# Spectrogram Generator
# ============================================================

def create_spectrogram(y, sr):

    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(np.abs(X), ref=np.max)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=64)

    ax.axis("off")

    librosa.display.specshow(
        Xdb,
        sr=sr,
        x_axis=None,
        y_axis=None,
        cmap="magma",
        ax=ax,
    )

    fig.canvas.draw()

    img = np.asarray(fig.canvas.buffer_rgba())
    img = img[:, :, :3]  # Remove alpha channel

    plt.close(fig)

    img = Image.fromarray(img)
    img = img.resize(TARGET_SIZE)

    return np.array(img)


# ============================================================
# Prediction Function
# ============================================================

def predict_emotion(audio_file, model, label_encoder, language):

    y, sr = librosa.load(audio_file, sr=22050)

    img = create_spectrogram(y, sr)

    x = img.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x, verbose=0)[0]

    idx = np.argmax(prediction)

    # Emotion labels
    if language == "English":
        classes = [
            "ANGRY",
            "NEUTRAL",
            "FEARFUL",
            "FEARFUL",
            "FEAR",
            "HAPPY",
            "NEUTRAL",
        ]
    else:  # Gujarati
        classes = [
            "ANGRY",
            "NEUTRAL",
            "FEARFUL",
            "FEARFUL",
            "FEAR",
            "HAPPY",
            "NEUTRAL",
        ]

    emotion = classes[idx]
    confidence = prediction[idx]

    return emotion, confidence, prediction, classes


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

            emotion, confidence, probabilities, classes = predict_emotion(
                uploaded_file,
                model,
                label_encoder,
                language)

        st.success(f"### Predicted Emotion : {emotion}")

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.markdown("## 📈 Prediction Probabilities")


        for cls, prob in zip(classes, probabilities):

            st.write(f"**{cls}**")

            st.progress(float(prob))

            st.write(f"{prob*100:.2f}%")
