import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os


# Chemin vers le modèle et les indices de classes
def get_model_and_classes():
    model_path = os.path.join(
        ".", "exports", "models", "densenet121_malaria_classifier.keras"
    )
    class_indices_path = os.path.join("..", "exports", "models", "class_indices.npy")
    model = load_model(model_path)
    if os.path.exists(class_indices_path):
        class_indices = np.load(class_indices_path, allow_pickle=True).item()
    else:
        # fallback: 0=Parasitized, 1=Uninfected
        class_indices = {"Parasitized": 0, "Uninfected": 1}
    classes = {v: k for k, v in class_indices.items()}
    return model, classes


def predict_image(model, classes, img_pil):
    img = img_pil.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    preds = model.predict(img_array, verbose=0)[0]
    class_idx = np.argmax(preds)
    class_name = classes[class_idx]
    confidence = preds[class_idx]
    return class_name, confidence, preds


st.set_page_config(page_title="Détection du paludisme - DenseNet121", layout="centered")
st.title("🦠 Détection de cellules infectées par le paludisme")
st.write(
    "Testez le modèle DenseNet121 entraîné sur vos propres images de cellules sanguines."
)

uploaded_file = st.file_uploader(
    "Choisissez une image de cellule sanguine (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Image chargée", use_container_width=True)
        model, classes = get_model_and_classes()
        class_name, confidence, preds = predict_image(model, classes, img)
        st.markdown(f"**Prédiction :** {class_name}")
        st.markdown(f"**Confiance :** {confidence*100:.2f}%")
        st.bar_chart({k: float(preds[v]) * 100 for v, k in classes.items()})
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
else:
    st.info("Veuillez charger une image pour commencer.")
