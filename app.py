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
from groq import Groq
import requests
from streamlit_js_eval import streamlit_js_eval

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY") or st.secrets.get("WEATHER_API_KEY", "")

if not GROQ_API_KEY:
    st.error("❌ Groq API key missing")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Plant Disease Detector", layout="wide")

BASE = Path(__file__).parent.resolve()
MODELS_DIR = BASE / "models"
IMAGES_DIR = BASE / "images"
IMAGE_SIZE = (128, 128)

# --------------------------------------------------
# LOCATION (GPS)
# --------------------------------------------------
coords = streamlit_js_eval(
    js_expressions='navigator.geolocation.getCurrentPosition((pos) => pos.coords)'
)

if coords:
    lat = coords["latitude"]
    lon = coords["longitude"]
else:
    lat, lon = None, None

# --------------------------------------------------
# WEATHER
# --------------------------------------------------
def get_weather(lat, lon):
    if not WEATHER_API_KEY or not lat:
        return None, None

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"

    res = requests.get(url).json()

    temp = res["main"]["temp"]
    humidity = res["main"]["humidity"]

    return temp, humidity

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def find_model_file(plant):
    p = MODELS_DIR / plant / "model.h5"
    if p.exists():
        return p
    alt = sorted((MODELS_DIR / plant).glob("*.h5"))
    return alt[0] if alt else None

def classmap_path_for(plant):
    return MODELS_DIR / plant / "class_map.json"

def plant_image_path(plant):
    for ext in (".jpg",".png",".jpeg"):
        p = IMAGES_DIR / f"{plant}{ext}"
        if p.exists():
            return p
    return None

def translate(text, lang):
    if lang == "en":
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

# --------------------------------------------------
# LANGUAGE
# --------------------------------------------------
LANGS = {
    "English": "en", "Hindi": "hi", "Bengali": "bn",
    "Tamil": "ta", "Telugu": "te"
}
selected_lang = st.sidebar.selectbox("Language", LANGS.keys())
lang_code = LANGS[selected_lang]

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model_for(plant):
    return load_model(str(find_model_file(plant)))

@st.cache_resource
def load_class_map_for(plant):
    with open(classmap_path_for(plant),"r") as f:
        d=json.load(f)
    return {v:k for k,v in d.items()}

# --------------------------------------------------
# GROQ AI
# --------------------------------------------------
def generate_agri_response(plant, disease, temp=None, humidity=None):
    weather_info = ""
    if temp and humidity:
        weather_info = f"Current weather: {temp}°C and {humidity}% humidity."

    prompt = f"""
You are an agricultural expert.

Plant: {plant}
Disease: {disease}
{weather_info}

Give structured scientific advice:

1. About Disease
2. Symptoms
3. Treatment (safe only)
4. Prevention
5. Nutrients
6. Fertilizers
7. Good Practices

No dosage. No banned chemicals.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a professional agricultural scientist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=800
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"❌ AI Error: {str(e)}"

# --------------------------------------------------
# HOME
# --------------------------------------------------
def home():
    st.title("🌿 Plant Disease Detector")

    plants = get_plants()
    cols = st.columns(4)

    for i, plant in enumerate(plants):
        with cols[i % 4]:
            imgp = plant_image_path(plant)
            if imgp:
                st.image(Image.open(imgp), width=150)

            if st.button(plant):
                st.session_state.page = "predict"
                st.session_state.plant = plant
                st.rerun()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
def prediction():
    plant = st.session_state.plant

    st.header(f"🍃 {plant} Detection")

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("### 📸 Capture or Upload Leaf Image")

    camera_img = st.camera_input("Take a photo")
    upload_img = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

    image_file = camera_img if camera_img else upload_img

    if image_file is None:
        return

    col1, col2 = st.columns(2)
    col1.image(image_file, width=250)

    if col2.button("🔍 Predict Disease"):
        with st.spinner("Analyzing..."):
            model = load_model_for(plant)
            inv_map = load_class_map_for(plant)

            img = kimage.load_img(image_file, target_size=IMAGE_SIZE)
            x = kimage.img_to_array(img)/255.0
            x = np.expand_dims(x,0)

            preds = model.predict(x)[0]
            idx = int(np.argmax(preds))

            disease = inv_map[idx].replace("_"," ")
            confidence = float(preds[idx]*100)

        st.success(f"{disease} ({confidence:.2f}%)")

        # Weather
        temp, humidity = get_weather(lat, lon)

        if temp:
            st.info(f"📍 Weather → 🌡 {temp}°C | 💧 {humidity}%")
        else:
            st.warning("⚠️ Weather not available")

        # AI Advice
        st.subheader("🛡 AI Advisory")

        with st.spinner("Generating advice..."):
            ai = generate_agri_response(plant, disease, temp, humidity)

        st.markdown(translate(ai, lang_code))

# --------------------------------------------------
# NAV
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

home() if st.session_state.page == "home" else prediction()