# HunyuanOCR – Tencent 1B OCR Expert

HunyuanOCR is a lightweight, high-performance OCR application powered by the **Tencent HunyuanOCR 1B model**. It supports document parsing, text detection, translation, and custom prompts. The system outputs tables as HTML, formulas as LaTeX, and text in a structured markdown format.

---

## Features

- Document Parsing: Extract all content from images in Markdown, with tables as HTML and formulas as LaTeX.
- Text Detection: Identify text and output coordinates.
- Translation: Extract text and translate to English.
- Custom Prompt: Flexible OCR tasks using user-defined prompts.
- Lightweight deployment using Streamlit.

---

## Requirements

- Python 3.10+
- Packages listed in `requirements.txt`:

```
streamlit>=1.38.0
torch>=2.3.0
pillow>=10.0.0
accelerate>=0.33.0
transformers@git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4
```

---

## Setup & Deployment

1. **Clone the repository**:

```bash
git clone https://github.com/ikantkode/hunyuan-1b-ocr-app.git
cd hunyuan-ocr-app
```

2. **Create and activate a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run the application**:

```bash
streamlit run app.py
```

5. **Load the model** from the sidebar (first-time setup may take a few minutes) and upload an image to start OCR.

---

## Folder Structure

```
hunyuan-ocr-app/
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── data/                 # Stores temporary files
└── hOCR/                 # (Optional) Additional project files, ignored in git
```

> Note: The `venv/` and `hOCR/` folders are ignored in Git and should not be pushed.

---

## Usage

1. Select the **language** (English or Chinese).
2. Choose a **task**: Document Parsing, Text Detection, Translation, or Custom Prompt.
3. Upload an image (PNG, JPG, JPEG, BMP, WEBP).
4. Click **Load Model** (if not loaded) and then **Run OCR**.
5. View results in the app and optionally download as `.txt`.

---

## License

This project is for personal and research use. The OCR model is provided by Tencent via Hugging Face: [tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR).

