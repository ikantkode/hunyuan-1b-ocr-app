# app.py — Final, Clean UI + Timer + Clear Backend Info
import streamlit as st
from PIL import Image
import torch
from pathlib import Path
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
import os
import tempfile
from pdf2image import convert_from_bytes
import time

# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

Path("./data").mkdir(exist_ok=True)

st.set_page_config(
    page_title="HunyuanOCR Pro",
    page_icon="magnifying glass",
    layout="centered",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS FOR CLEAN LOOK ===
st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    .stSpinner > div {border-top-color: #1E88E5 !important;}
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        background: #f8f9fa;
        border-left: 5px solid #1E88E5;
        margin-top: 1rem;
    }
    .timer {
        font-size: 1.1rem;
        color: #1565C0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# === SESSION STATE ===
for k in ["model_loaded", "processor", "model"]:
    if k not in st.session_state:
        st.session_state[k] = None

# === HELPER FUNCTIONS ===
def clean_repeated(text: str) -> str:
    n = len(text)
    if n < 8000: return text
    for l in range(2, n // 10 + 1):
        cand = text[-l:]
        cnt = 0
        i = n - l
        while i >= 0 and text[i:i+l] == cand:
            cnt += 1
            i -= l
        if cnt >= 10:
            return text[:n - l * (cnt - 1)]
    return text

def resize_image(img: Image.Image, max_pixels=1048576) -> Image.Image:
    if img.width * img.height > max_pixels:
        ratio = (max_pixels / (img.width * img.height)) ** 0.5
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img

def load_model():
    if st.session_state.model_loaded:
        return True
    with st.spinner("Loading HunyuanOCR 1B model (Transformers + Accelerate)..."):
        try:
            processor = AutoProcessor.from_pretrained("tencent/HunyuanOCR", use_fast=False)
            model = HunYuanVLForConditionalGeneration.from_pretrained(
                "tencent/HunyuanOCR",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            st.session_state.processor = processor
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.success("Model loaded successfully!")
            return True
        except Exception as e:
            st.error("Failed to load model")
            st.exception(e)
            return False

def process_image(img: Image.Image) -> str:
    img = resize_image(img)
    prompt = ("Extract all information from the main body of the document image and represent it in markdown format, "
              "ignoring headers and footers. Tables should be expressed in HTML format, formulas in LaTeX, "
              "and the parsing should be organized according to the reading order.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        img.save(f.name, quality=92)
        img_path = f.name

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": prompt}
    ]}]

    text = st.session_state.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = st.session_state.processor(text=text, images=img, return_tensors="pt")
    inputs = {k: v.to(st.session_state.model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = st.session_state.model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=False,
            temperature=0.0
        )

    generated = generated[:, inputs["input_ids"].shape[1]:]
    result = st.session_state.processor.batch_decode(generated, skip_special_tokens=True)[0]
    os.unlink(img_path)
    return clean_repeated(result)

# === SIDEBAR ===
with st.sidebar:
    st.header("HunyuanOCR Pro")
    st.markdown("**Best-in-class 1B OCR model**")
    
    if not st.session_state.model_loaded:
        if st.button("Load Model (one-time)", type="primary", use_container_width=True):
            load_model()
    else:
        st.success("Model Ready")
        if st.button("Free VRAM", use_container_width=True):
            st.session_state.model = None
            st.session_state.processor = None
            st.session_state.model_loaded = False
            torch.cuda.empty_cache()
            st.rerun()

    st.markdown("---")
    st.caption("Inference Engine: **Transformers + Accelerate** (not vLLM)")

# === MAIN UI ===
st.title("HunyuanOCR Pro")
st.markdown("### High-Accuracy Document & PDF → Clean Markdown")

st.info("""
**Features**  
- Multi-page PDF support  
- Tables → HTML • Math → LaTeX  
- No OOM crashes (auto-resize)  
- Works on 12GB+ GPUs  
- Powered by **Transformers + Accelerate** backend
""")

uploaded = st.file_uploader(
    "Upload Image or PDF",
    type=["png", "jpg", "jpeg", "webp", "pdf"],
    help="Supports single images and multi-page PDFs"
)

if uploaded and st.session_state.model_loaded:
    # Load pages
    if uploaded.type == "application/pdf":
        pages = convert_from_bytes(uploaded.read(), dpi=160)
        st.success(f"Loaded {len(pages)} page(s)")
    else:
        pages = [Image.open(uploaded).convert("RGB")]

    # Preview thumbnails
    cols = st.columns(min(len(pages), 5))
    for i, page in enumerate(pages[:5]):
        with cols[i]:
            st.image(page, caption=f"Page {i+1}", width=120)
    if len(pages) > 5:
        st.caption(f"... and {len(pages)-5} more pages")

    if st.button("Run OCR → Markdown", type="primary", use_container_width=True):
        all_results = []
        timer = st.empty()
        progress = st.progress(0)
        start_time = time.time()

        for i, page in enumerate(pages):
            timer.markdown(f"<div class='timer'>Processing page {i+1}/{len(pages)} • Elapsed: {time.time() - start_time:.1f}s</div>", unsafe_allow_html=True)
            result = process_image(page)
            all_results.append(f"# Page {i+1}\n\n{result}\n\n---\n")
            progress.progress((i + 1) / len(pages))

        total_time = time.time() - start_time
        final_md = "\n".join(all_results)

        progress.empty()
        timer.empty()

        st.success(f"OCR Complete in {total_time:.1f} seconds!")

        st.markdown("### Result")
        with st.container():
            st.markdown(f"<div class='result-box'>{final_md}</div>", unsafe_allow_html=True)

        st.download_button(
            label="Download Full Markdown",
            data=final_md,
            file_name=f"hunyuanocr_{uploaded.name.split('.')[0]}.md",
            mime="text/markdown",
            use_container_width=True
        )
else:
    if uploaded:
        st.warning("Please load the model first (sidebar)")
    else:
        st.info("Upload a document or PDF to begin")

st.markdown("---")
st.caption("Made with ❤️ using [tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) • Transformers backend")