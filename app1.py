import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io
import os

st.set_page_config(page_title="🖼️ Text-to-Image Generator", layout="centered")
st.title("🧠 Hugging Face Text-to-Image Generator")

# Input Prompt
prompt = st.text_input("Enter your prompt", value="A magical forest with glowing animals")

# Model Selection (Optional)
model_id = "stabilityai/stable-diffusion-3"

# Generate Button
if st.button("🎨 Generate Image") and prompt:
    with st.spinner("Generating image..."):
        try:
            # Load HF token from secrets
            HF_TOKEN = st.secrets["HF_TOKEN"]

            # Init client
            client = InferenceClient(token=HF_TOKEN)

            # Generate image
            image = client.text_to_image(prompt=prompt, model=model_id)

            # Display
            st.image(image, caption="Generated Image", use_column_width=True)

            # Download
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="📥 Download Image",
                data=buf.getvalue(),
                file_name="image.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"🛑 Failed: {e}")


