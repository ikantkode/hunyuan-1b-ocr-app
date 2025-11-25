# app.py
import streamlit as st
from PIL import Image
import torch
from pathlib import Path
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
import os
import tempfile

# Create data folder
Path("./data").mkdir(exist_ok=True)

# Page config
st.set_page_config(
    page_title="HunyuanOCR – Tencent 1B OCR Expert",
    page_icon="magnifying glass",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
for key in ["model_loaded", "processor", "model", "lang"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "lang" not in st.session_state:
    st.session_state.lang = "english"

# ------------------------------------------------------------------
# Critical fix: clean repeated substrings (from official model card)
# ------------------------------------------------------------------
def clean_repeated_substrings(text: str) -> str:
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[:n - length * (count - 1)]
    return text

# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
def load_model():
    if st.session_state.model_loaded:
        return True

    with st.spinner("Loading HunyuanOCR 1B model (first time only, ~2GB)..."):
        try:
            model_name = "tencent/HunyuanOCR"

            processor = AutoProcessor.from_pretrained(model_name, use_fast=False)

            model = HunYuanVLForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

            st.session_state.processor = processor
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.success("HunyuanOCR loaded successfully!")
            return True
        except Exception as e:
            st.error("Failed to load model")
            st.exception(e)
            return False

# ------------------------------------------------------------------
# Inference function – THE ONE THAT ACTUALLY WORKS
# ------------------------------------------------------------------
def infer(image: Image.Image, task: str, custom_prompt: str = None):
    # Prompt dictionary (official prompts from model card)
    prompts = {
        "parsing": {
            "english": "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order.",
            "chinese": "提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。"
        },
        "spotting": {
            "english": "Detect and recognize text in the image, and output the text coordinates in a formatted manner.",
            "chinese": "检测并识别图片中的文字，将文本坐标格式化输出。"
        },
        "translation": {
            "english": "First extract the text, then translate the text content into English. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format.",
            "chinese": "先提取文字，再将文字内容翻译为英文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。"
        },
        "custom": {
            "english": custom_prompt or "Extract and process the text from this image.",
            "chinese": custom_prompt or "从这张图片中提取并处理文本。"
        }
    }

    prompt_text = prompts[task][st.session_state.lang]

    # Save image to temp file (required by chat template)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        image.save(f.name)
        img_path = f.name

    # Build message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    # Apply chat template
    text = st.session_state.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    inputs = st.session_state.processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(st.session_state.model.device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            generated_ids = st.session_state.model.generate(
                **inputs,
                max_new_tokens=16384,
                do_sample=False,
                temperature=0.0
            )

        # Trim prompt tokens
        input_ids = inputs["input_ids"]
        generated_ids = generated_ids[:, input_ids.shape[1]:]

        # Decode
        result = st.session_state.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        os.unlink(img_path)
        return clean_repeated_substrings(result)

    except Exception as e:
        if os.path.exists(img_path):
            os.unlink(img_path)
        return f"Inference error: {e}"

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    st.session_state.lang = st.selectbox("Language", ["english", "chinese"], index=0 if st.session_state.lang == "english" else 1)

    task = st.selectbox(
        "Task",
        options=["parsing", "spotting", "translation", "custom"],
        format_func=lambda x: {
            "parsing": "Document Parsing (Markdown + HTML + LaTeX)",
            "spotting": "Text Detection + Coordinates",
            "translation": "Extract → Translate to English",
            "custom": "Custom Prompt"
        }[x]
    )

    custom_prompt = None
    if task == "custom":
        custom_prompt = st.text_area("Custom Prompt", height=150)

    if not st.session_state.model_loaded:
        if st.button("Load Model (one-time)", type="primary", use_container_width=True):
            load_model()
    else:
        st.success("Model ready")
        if st.button("Free VRAM (unload)", use_container_width=True):
            st.session_state.model_loaded = False
            st.session_state.model = None
            st.session_state.processor = None
            torch.cuda.empty_cache()
            st.rerun()

# ------------------------------------------------------------------
# Main UI
# ------------------------------------------------------------------
st.title("HunyuanOCR – Tencent 1B OCR Expert")
st.caption("Best-in-class lightweight OCR • Tables → HTML • Formulas → LaTeX • Multilingual")

uploaded_file = st.file_uploader(
    "Upload an image (document, receipt, screenshot, etc.)",
    type=["png", "jpg", "jpeg", "bmp", "webp"]
)

if uploaded_file and st.session_state.model_loaded:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("Run OCR", type="primary", use_container_width=True):
            with st.spinner("Processing image with HunyuanOCR..."):
                result = infer(image, task, custom_prompt)
            st.markdown("### Result")
            st.markdown(result)

            st.download_button(
                label="Download Result",
                data=result,
                file_name=f"hunyuanocr_{task}_{st.session_state.lang}.txt",
                mime="text/plain",
                use_container_width=True
            )
else:
    if uploaded_file:
        st.info("Click **Load Model** in the sidebar first.")
    else:
        st.info("Upload an image to get started.")

st.markdown("---")
st.markdown("Powered by **[tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR)** • Built with Streamlit • Transformers + Accelerate")