import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from PIL import Image
import json

# Configure the Streamlit page
st.set_page_config(page_title="SnapTeller", page_icon="📸", layout="centered")

# Load model
@st.cache_resource
def load_caption_model():
    return load_model("my_model_2.keras")

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.json", "r") as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer

# Convert index to word
def idx_to_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None

# Extract image features using VGG16
def extract_features(image):
    model = VGG16()
    model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = image.resize((224, 224))
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    feature = model.predict(image_array, verbose=0)
    return feature

# Generate caption from image
def generate_caption(image, model, tokenizer, max_length=35):
    features = extract_features(image)
    in_text = 'startseq'
    
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding='post')
        yhat = model.predict([features, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break

    return in_text.replace('startseq', '').replace('endseq', '').strip()

# ------------------------------------------
# 🖼️ SnapTeller App UI

# Title section
st.markdown("""
<div style='text-align:center'>
    <h1 style='color:#FF4B4B; font-size:52px;'>📸 SnapTeller</h1>
    <p style='font-size:20px; color:gray;'>Let your image speak its story</p>
</div>
<hr style='margin-bottom: 30px;' />
""", unsafe_allow_html=True)

# Upload section
uploaded_image = None
image_uploaded = False

upload_btn = st.file_uploader("📤 Click below to upload your image", type=["jpg", "jpeg", "png"], key="uploader")

if upload_btn is not None:
    uploaded_image = Image.open(upload_btn)
    st.image(uploaded_image, caption="✅ Image Uploaded", use_container_width=True)
    st.markdown("<div style='text-align: center; font-size: 28px; color: #FF4B4B; margin-top: 10px;'>SnapTeller</div>", unsafe_allow_html=True)
    st.markdown("---")
    image_uploaded = True

# Generate caption button (only shown after upload)
if image_uploaded:
    if st.button("✨ Generate Caption"):
        with st.spinner("🔍 Analyzing image... Generating caption..."):
            model = load_caption_model()
            tokenizer = load_tokenizer()
            caption = generate_caption(uploaded_image, model, tokenizer)

        st.success("✅ Caption Generated!")

        st.markdown(f"""
        <div style='background-color: #f0f8ff; padding: 20px; border-radius: 12px; text-align: center;'>
            <h3 style='color: #1ABC9C;'>📝 Caption:</h3>
            <p style='font-size: 22px; color: #333; font-style: italic;'>
                "{caption}"
            </p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("👆 Please upload an image to enable caption generation.")
