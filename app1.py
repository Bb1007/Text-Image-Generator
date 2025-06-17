import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import os

# Sidebar token & model config
HF_TOKEN = os.getenv("HF_TOKEN")
model_id = "runwayml/stable-diffusion-v1-5"

# Cache the pipeline init (only once!)
@st.cache_resource(show_spinner=False)
def load_model():
    return StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_auth_token=HF_TOKEN
    ).to("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="AI Image Gen", layout="wide")
st.title("🖼️ Text‑to‑Image Generator")

prompt = st.text_input("Enter your prompt:", "")
if st.button("Generate"):
    pipe = load_model()
    with st.spinner("Generating…"):
        img = pipe(prompt).images[0]
    st.image(img, caption=prompt, use_column_width=True)
    img.save("output.png")
    st.download_button("Download Image", data=open("output.png","rb"), file_name="output.png", mime="image/png")
