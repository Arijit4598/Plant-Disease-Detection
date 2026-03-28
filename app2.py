import streamlit as st
from groq import Groq

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Agricultural Disease Advisory AI",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Agricultural Disease Advisory System")
st.write(
    "Scientifically accurate, safe, and extension-grade crop disease guidance "
    "powered by Groq LLM."
)

# --------------------------------------------------
# API Setup
# --------------------------------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("❌ Groq API key not found. Add GROQ_API_KEY in secrets.toml")
    st.stop()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --------------------------------------------------
# Generation Function
# --------------------------------------------------
def generate_agri_response(plant, disease):
    prompt = f"""
You are a highly experienced agricultural scientist and plant pathologist.

Provide scientifically accurate, field-level actionable advice.

Crop: {plant}
Disease/Problem: {disease}

Follow this STRICT structure:

### 1. About the Disease
(Include pathogen type: fungal/bacterial/viral)

### 2. Symptoms
- Leaves
- Stems
- Roots
- Fruits (if applicable)

### 3. Safe & Legal Treatment Options
(Use only approved fungicides & practices, no dosage)

### 4. Prevention
(Field-level practices)

### 5. Nutrient Requirements
(N, P, K, micronutrients explained clearly)

### 6. Fertilizer Recommendations
(Organic + chemical + biofertilizers)

### 7. Additional Good Practices

Make response practical for farmers in India.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # ⚡ super fast
        messages=[
            {"role": "system", "content": "You are a professional agricultural advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )

    return response.choices[0].message.content

# --------------------------------------------------
# UI Inputs
# --------------------------------------------------
with st.form("agri_form"):
    plant = st.text_input("🌾 Crop / Plant Name", placeholder="e.g. Potato")
    disease = st.text_input("🦠 Disease / Problem", placeholder="e.g. Late Blight")
    submitted = st.form_submit_button("Generate Advisory")

# --------------------------------------------------
# Output
# --------------------------------------------------
if submitted:
    if not plant or not disease:
        st.warning("Please enter both plant and disease.")
    else:
        with st.spinner("Generating expert guidance..."):
            result = generate_agri_response(plant, disease)

        st.markdown("## 📋 Advisory Report")
        st.markdown(result)