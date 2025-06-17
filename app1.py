import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io

st.set_page_config(page_title="🖼️ Text to Image Generator", layout="centered")
st.title("🧠 Hugging Face Text → Image")

huggingface_token = st.secrets["HUGGINGFACE_TOKEN"]

@st.cache_resource
def load_pipeline():
    return StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        use_auth_token=huggingface_token,
        revision="fp16"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

pipe = load_pipeline()

prompt = st.text_input("Enter your prompt:")
if st.button("Generate") and prompt:
    with st.spinner("Generating image..."):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        st.download_button("📥 Download", data=img_byte_arr.getvalue(), file_name="image.png", mime="image/png")

