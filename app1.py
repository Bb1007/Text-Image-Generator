# app.py
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os

# ----- APP UI SETUP -----
st.set_page_config(page_title="🧠 Text-to-Image Generator", layout="centered")
st.title("🎨 Hugging Face Text-to-Image Generator")
st.write("Type a prompt and generate AI art using Stable Diffusion.")

# ----- USER INPUT -----
prompt = st.text_input("Enter your image prompt:", placeholder="e.g. A futuristic city at sunset")

# ----- HUGGING FACE API TOKEN -----
HF_TOKEN = st.secrets.get("HF_TOKEN")  # Securely use Hugging Face token from Streamlit secrets

# Model Endpoint
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# ----- CALL API -----
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"🛑 Failed to generate image. Error {response.status_code}: {response.text}")
        return None

# ----- GENERATE IMAGE -----
if st.button("🎨 Generate Image"):
    if prompt:
        with st.spinner("Generating..."):
            image_bytes = query({"inputs": prompt})
            if image_bytes:
                image = Image.open(BytesIO(image_bytes))
                st.image(image, caption="Generated Image", use_column_width=True)
                st.success("Image Generated!")
    else:
        st.warning("Please enter a prompt first.")



