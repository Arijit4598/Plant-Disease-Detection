import streamlit as st
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as kimage
from deep_translator import GoogleTranslator  
from dotenv import load_dotenv
import os
import requests

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


st.set_page_config(page_title="Plant Disease Detector", layout="wide")

BASE = Path(__file__).parent.resolve()
MODELS_DIR = BASE / "models"
IMAGES_DIR = BASE / "images"
IMAGE_SIZE = (128, 128)

st.markdown("""
<style>
header{visibility:hidden;height:0}
section.main>div{padding-top:0!important}
.block-container{padding-top:1rem!important;padding-bottom:0!important;box-shadow:none!important}
.stButton>button{ white-space: nowrap; }
main{background:transparent!important;overflow-x:hidden}
body{
    background:linear-gradient(135deg,#020612,#03121f,#041d26);
    margin:0;
}
.page-title{
    text-align:center;
    font-size:50px;
    font-weight:900;
    color:#62ffb8;
    text-shadow:0 0 10px #62ffb8,0 0 30px #62ffb8;
    animation:glow 2s ease-in-out infinite alternate;
    margin-bottom:5px;
}
@keyframes glow{
    from{text-shadow:0 0 10px #62ffb8,0 0 20px #62ffb8}
    to{text-shadow:0 0 25px #62ffb8,0 0 50px #62ffb8}
}
.subtitle{
    text-align:center;
    color:#b7ffd8;
    font-size:20px;
    margin-top:-5px;
    margin-bottom:40px;
}
.card-box{
    backdrop-filter:blur(10px);
    background:rgba(255,255,255,0.06);
    border-radius:18px;
    padding:18px 18px 28px 18px;
    text-align:center;
    border:1px solid rgba(98,255,184,0.25);
    box-shadow:0 0 15px rgba(98,255,184,0.12);
    transition:0.25s;
}
.card-box:hover{
    transform:scale(1.07);
    box-shadow:0 0 40px rgba(98,255,184,0.45);
}
.card-img{
    width:140px;
    border-radius:14px;
    box-shadow:0 0 12px #62ffb840;
    margin-bottom:12px;
}
.stButton>button{
    background:linear-gradient(135deg,#0f806d,#04ffb4);
    color:#00110c;
    font-weight:700;
    min-width:140px;
    max-width:max-content;
    margin:auto;
    border:none;
    border-radius:10px;
    box-shadow:none;
    transition:0.3s;
}
.stButton>button:hover{
    transform:scale(1.1);
    box-shadow:0 0 22px #62ffb8;
}
.upload-box{
    border:2px dashed #00ffc6;
    padding:25px;
    border-radius:15px;
    text-align:center;
    color:#74ffd0;
    margin-top:10px;
    backdrop-filter:blur(8px);
    background:rgba(255,255,255,0.05);
}
.result-box{
    background:rgba(255,255,255,0.07);
    padding:18px;
    border-radius:12px;
    color:#00ffc6;
    font-size:24px;
    font-weight:700;
    text-align:center;
    width:100%;
    backdrop-filter:blur(8px);
    border:1px solid rgba(98,255,184,0.25);
    box-shadow:0 0 30px rgba(98,255,184,0.25);
    margin-top:15px;
    animation:pop 0.3s ease-out;
}
@keyframes pop{
    from{transform:scale(0.8);opacity:0}
    to{transform:scale(1);opacity:1}
}
.st-emotion-cache-znj1k1{
    display:none;
}
.center-row{
    display:flex;
    width:100%;
    height:auto;
    justify-content:center;
    align-items:center;
    gap:30px;
    margin-top:20px;
}
.st-emotion-cache-1permvm{
justify-content:center;
align-items:center;
gap:100px;
}
            .st-emotion-cache-1wpb1x8 {
       width: auto;
    flex: none;
}
</style>
""", unsafe_allow_html=True)

def find_model_file(plant):
    p = MODELS_DIR / plant / "model.h5"
    if p.exists():
        return p
    alt = sorted((MODELS_DIR / plant).glob("*.h5"))
    return alt[0] if alt else None

def classmap_path_for(plant):
    return MODELS_DIR / plant / "class_map.json"


def plant_image_path(plant):
    for ext in (".jpg",".jpeg",".png"):
        p = IMAGES_DIR / f"{plant}{ext}"
        if p.exists():
            return p
    return None

def load_prevention_map(plant):
    path = MODELS_DIR / plant / f"{plant}_prevention.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def translate(text, lang):
    if lang == "en" or not text:
        return text
    try:
        return GoogleTranslator(source="en", target=lang).translate(text)
    except:
        return text

def get_plants():
    a=[]
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.iterdir()):
            if p.is_dir() and classmap_path_for(p.name).exists() and find_model_file(p.name):
                a.append(p.name)
    return a

LANGS = {
    "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr",
    "German": "de", "Bengali": "bn", "Tamil": "ta", "Telugu": "te"
}
selected_lang = st.sidebar.selectbox("Select Language", LANGS.keys())
lang_code = LANGS[selected_lang]


@st.cache_resource
def load_model_for(plant):
    return load_model(str(find_model_file(plant)))

@st.cache_resource
def load_class_map_for(plant):
    with open(classmap_path_for(plant),"r") as f:
        d=json.load(f)
    return {v:k for k,v in d.items()}

def home():
    st.markdown('<div class="page-title">🌿 Plant Disease Detector</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Choose a plant to start prediction</p>', unsafe_allow_html=True)

    plants = get_plants()
    if not plants:
        st.error("No valid plants found in /models")
        return

    cols_per_row = 4

    for i in range(0, len(plants), cols_per_row):
        row_plants = plants[i:i + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, plant in zip(cols, row_plants):
            with col:
                st.markdown('<div class="card-box">', unsafe_allow_html=True)

                imgp = plant_image_path(plant)
                if imgp:
                    st.image(Image.open(imgp), width=180)

                if st.button(plant, key=f"btn_{plant}"):
                    st.session_state.page = "predict"
                    st.session_state.plant = plant
                    st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)


def generate_agri_response(plant, disease):
    prompt = f"""
You are an agricultural expert specializing in plant pathology, crop nutrition, and safe farm management.

Plant: {plant}
Issue: {disease}

Your response MUST follow this structure:

### 1. About the Disease
### 2. Symptoms
### 3. Safe & Legal Treatment Options
### 4. Prevention
### 5. Nutrient Requirements
### 6. Fertilizer Recommendations
### 7. Additional Good Practices
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Agricultural Disease Advisory AI"
        },
        json={
            "model": "mistralai/mistral-small-3.1-24b-instruct:free",
            "messages": [
                {"role": "system", "content": "You are a strict agricultural science expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 900
        },
        timeout=60
    )

    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

def prediction():
    plant = st.session_state.plant

    st.markdown(
        f'<div class="page-title">🍃 {plant} Detection</div>',
        unsafe_allow_html=True
    )

    # Back button
    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

    # Upload section
    st.markdown(
        '<div class="upload-box">Upload a leaf image</div>',
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if not uploaded:
        return

    # Layout
    col1, col2 = st.columns(2)
    col1.image(uploaded, width=240)

    # Predict button
    if col2.button("🔍 Predict Disease"):

        with st.spinner("Analyzing leaf image..."):
            # Load model & class map
            model = load_model_for(plant)
            inv_map = load_class_map_for(plant)

            # Preprocess image
            img = kimage.load_img(uploaded, target_size=IMAGE_SIZE)
            x = kimage.img_to_array(img) / 255.0
            x = np.expand_dims(x, 0)

            # Prediction
            preds = model.predict(x)[0]
            idx = int(np.argmax(preds))

            raw_disease = inv_map[idx]
            disease = raw_disease.replace("_", " ")
            confidence = float(preds[idx] * 100)

        # Result display
        st.markdown(
            f'<div class="result-box">✅ {plant} → {disease} ({confidence:.2f}%)</div>',
            unsafe_allow_html=True
        )

        # AI-powered Prevention & Cure
        st.subheader("🛡 Prevention, Cure & Crop Management")

        with st.spinner("Scientific agricultural advice..."):
            ai_response = generate_agri_response(
                plant=plant,
                disease=disease
            )

        ai_response_translated = translate(ai_response, lang_code)

        st.markdown(ai_response_translated)

if "page" not in st.session_state:
    st.session_state.page = "home"

home() if st.session_state.page == "home" else prediction()
