import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Text-to-Image Generator")
st.title("🎨 AI Image Generator with Hugging Face")
st.write("Type a prompt and generate an image using Stable Diffusion!")

prompt = st.text_input("Enter your prompt:")

def query(payload):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_TOKEN']}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

if st.button("Generate Image") and prompt:
    with st.spinner("Generating..."):
        image_bytes = query({"inputs": prompt})
        image = Image.open(BytesIO(image_bytes))
        st.image(image, caption="Generated Image", use_column_width=True)
        st.download_button("📥 Download Image", image_bytes, file_name="generated.png")


