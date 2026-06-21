import io
import json
import math
import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import speech_recognition as sr
import streamlit as st
from gtts import gTTS

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
except Exception as exc:
    tf = None
    image = None
    TENSORFLOW_IMPORT_ERROR = exc
else:
    TENSORFLOW_IMPORT_ERROR = None

from database import authenticate_user, count_yield_records, create_user, import_sample_data
from i18n import (
    CROP_NAMES,
    LANGUAGES,
    SOIL_NAMES,
    TTS_LANG_CODES,
    canonical_label,
    tr,
    translate_crop,
    translate_soil,
)


FALLBACK_UI_TEXT = {
    "tab_yield": "Yield Prediction",
    "tab_classifier": "Classifier",
    "tab_fertilizer": "Fertilizer",
    "tab_chatbot": "Agri Chatbot",
    "tab_database": "Database",
}


def readable_key(key):
    text = key.removeprefix("tab_").replace("_", " ")
    return text.title()


def ui_text(language, key):
    try:
        value = tr(language, key)
        if value != key:
            return value
    except KeyError:
        pass

    try:
        value = tr("en", key)
        if value != key:
            return value
    except KeyError:
        pass

    return FALLBACK_UI_TEXT.get(key, readable_key(key))


BASE_DIR = Path(__file__).resolve().parent
YIELD_MODEL_PATH = BASE_DIR / "crop_yield_model.pkl"
SOIL_MODEL_PATH = BASE_DIR / "soil_model.h5"
CROP_MODEL_PATH = BASE_DIR / "crop_model.h5"
SOIL_CLASSES_PATH = BASE_DIR / "soil_classes.json"
CROP_CLASSES_PATH = BASE_DIR / "crop_classes.json"
SOIL_DATASET_DIR = BASE_DIR / "dataset" / "soil"
CROP_DATASET_DIR = BASE_DIR / "dataset" / "crop"
HERO_IMAGE_PATH = CROP_DATASET_DIR / "rice" / "rice-fields-204128_1280.jpg"

FEATURE_COLUMNS = [
    "temperature_c",
    "rainfall_mm",
    "irrigation_mm",
    "soil_n",
    "soil_p",
    "soil_k",
    "fertilizer_kg",
]

FERTILIZER_DATA = {
    "Wheat": {"Loam Soil": {"N": 258, "P": 188, "K": 65.8}},
    "Maize": {"Loam Soil": {"N": 350, "P": 200, "K": 120}},
}

BAG_WEIGHT_KG = 50
UREA_N_PERCENT = 0.46
DAP_P_PERCENT = 0.46
MOP_K_PERCENT = 0.60
MRP_UREA = 268
MRP_DAP = 1350
MRP_MOP = 800

CHATBOT_DISCLAIMER = (
    "Confirm chemical doses, pesticides, and disease treatment with a local "
    "agriculture officer or soil-test lab."
)

AGRI_TOPICS = {
    "yield": {
        "keywords": ["yield", "production", "harvest", "output", "increase"],
        "answer": (
            "To improve yield, start with a soil test, choose a crop suited to the "
            "season, keep irrigation steady during flowering and grain filling, "
            "and apply nutrients in split doses."
        ),
    },
    "fertilizer": {
        "keywords": [
            "fertilizer",
            "npk",
            "urea",
            "dap",
            "mop",
            "nutrient",
            "nitrogen",
            "phosphorus",
            "potassium",
        ],
        "answer": (
            "Use fertilizer based on soil-test values and crop stage. Nitrogen "
            "supports leafy growth, phosphorus helps roots, and potassium improves "
            "stress tolerance and grain or fruit quality."
        ),
    },
    "irrigation": {
        "keywords": ["water", "irrigation", "rainfall", "drought", "moisture"],
        "answer": (
            "Irrigate when the root zone is drying, not only by calendar date. "
            "Critical stages are germination, flowering, and grain or fruit filling."
        ),
    },
    "soil": {
        "keywords": ["soil", "ph", "loamy", "clay", "sandy", "black", "red", "organic"],
        "answer": (
            "Healthy soil should drain well, hold moisture, and contain organic "
            "matter. Add compost or farmyard manure where possible and test pH and "
            "NPK before major fertilizer decisions."
        ),
    },
    "pest": {
        "keywords": ["pest", "insect", "disease", "fungus", "leaf", "spots", "wilting", "yellow"],
        "answer": (
            "For pest or disease issues, first identify the symptom pattern: leaf "
            "spots, chewing, wilting, yellowing, or stem damage. Remove heavily "
            "infected plant parts and avoid overhead watering for fungal issues."
        ),
    },
    "crop": {
        "keywords": ["crop", "rice", "wheat", "maize", "cotton", "sugarcane", "sowing", "plant"],
        "answer": (
            "Crop choice should match season, soil, water availability, and market "
            "access. Rice needs more water, wheat prefers cool dry weather, maize "
            "needs good drainage, cotton suits black soil, and sugarcane needs a "
            "long warm season."
        ),
    },
}

QUICK_QUESTIONS = [
    "How can I improve crop yield?",
    "Which fertilizer should I use?",
    "How often should I irrigate?",
    "How do I improve soil health?",
    "What should I do for pest damage?",
]


def image_to_data_uri(path):
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def apply_custom_theme():
    hero_image = image_to_data_uri(HERO_IMAGE_PATH)
    st.markdown(
        f"""
        <style>
            :root {{
                --leaf: #1f7a4d;
                --leaf-dark: #0f3d2b;
                --soil: #8a5a32;
                --sky: #eaf7ff;
                --mint: #e7f7ee;
                --ink: #173025;
            }}

            .stApp {{
                background:
                    linear-gradient(135deg, rgba(240, 249, 242, 0.94), rgba(234, 247, 255, 0.92)),
                    url("{hero_image}");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
                color: #10281e;
            }}

            [data-testid="stSidebar"] {{
                background: rgba(255, 255, 255, 0.92);
                border-right: 1px solid rgba(31, 122, 77, 0.16);
            }}

            .block-container {{
                padding-top: 2rem;
                padding-bottom: 3rem;
            }}

            h1, h2, h3 {{
                color: #0b5d3a;
                font-weight: 800;
            }}

            [data-testid="stMarkdownContainer"] p {{
                color: #24483a;
            }}

            [data-testid="stWidgetLabel"] label,
            [data-testid="stTextInput"] label,
            [data-testid="stNumberInput"] label,
            [data-testid="stSelectbox"] label,
            [data-testid="stRadio"] label {{
                color: #103527;
                font-weight: 700;
            }}

            div[data-testid="stMetric"],
            div[data-testid="stForm"],
            div[data-testid="stExpander"] {{
                border: 1px solid rgba(31, 122, 77, 0.16);
                box-shadow: 0 14px 40px rgba(15, 61, 43, 0.08);
            }}

            .login-hero {{
                min-height: 72vh;
                display: grid;
                align-items: center;
            }}

            .brand-panel {{
                min-height: 520px;
                border-radius: 8px;
                padding: 38px;
                position: relative;
                overflow: hidden;
                background:
                    linear-gradient(145deg, rgba(6, 30, 21, 0.9), rgba(22, 91, 58, 0.7)),
                    url("{hero_image}");
                background-size: cover;
                background-position: center;
                color: white;
                box-shadow: 0 24px 70px rgba(15, 61, 43, 0.28);
            }}

            .brand-panel,
            .brand-panel div,
            .brand-panel p,
            .brand-panel span,
            .brand-panel strong {{
                color: #ffffff !important;
            }}

            .brand-kicker {{
                display: inline-flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 18px;
                color: #dfffee !important;
                font-size: 0.84rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-shadow: 0 2px 12px rgba(0, 0, 0, 0.58);
            }}

            .leaf-mark {{
                width: 22px;
                height: 22px;
                display: inline-block;
                border-radius: 22px 0 22px 0;
                background: linear-gradient(135deg, #d9ff66, #56d483);
                box-shadow: 0 0 0 5px rgba(202, 255, 223, 0.14);
                transform: rotate(-16deg);
            }}

            .brand-panel h1 {{
                color: #ffffff !important;
                font-size: clamp(2.2rem, 4vw, 4.3rem);
                line-height: 1.02;
                margin: 0 0 18px;
                font-weight: 900;
                text-shadow: 0 5px 26px rgba(0, 0, 0, 0.72);
            }}

            .brand-panel p {{
                max-width: 640px;
                font-size: 1.06rem;
                line-height: 1.65;
                color: rgba(255, 255, 255, 0.98) !important;
                text-shadow: 0 2px 14px rgba(0, 0, 0, 0.58);
            }}

            .signal-grid {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                margin-top: 34px;
            }}

            .signal-card {{
                border: 1px solid rgba(223, 255, 238, 0.46);
                border-radius: 8px;
                padding: 16px;
                background: rgba(4, 29, 20, 0.58);
                backdrop-filter: blur(12px);
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.16);
            }}

            .signal-card strong {{
                display: block;
                font-size: 1.5rem;
                color: #ffffff !important;
                text-shadow: 0 2px 12px rgba(0, 0, 0, 0.56);
            }}

            .signal-card span {{
                color: #f1fff7 !important;
                font-weight: 700;
                text-shadow: 0 2px 12px rgba(0, 0, 0, 0.56);
            }}

            .trust-strip {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 18px;
                max-width: 760px;
            }}

            .trust-pill {{
                border-radius: 999px;
                padding: 9px 14px;
                color: #ffffff !important;
                background: rgba(2, 18, 12, 0.78);
                border: 1px solid rgba(202, 255, 223, 0.58);
                font-size: 0.92rem;
                font-weight: 800;
                text-shadow: 0 2px 10px rgba(0, 0, 0, 0.7);
            }}

            .ai-band {{
                position: static;
                width: min(420px, 100%);
                border-radius: 8px;
                padding: 18px;
                margin-top: 18px;
                background:
                    linear-gradient(90deg, rgba(255,255,255,0.18) 1px, transparent 1px),
                    linear-gradient(rgba(255,255,255,0.16) 1px, transparent 1px),
                    rgba(3, 18, 13, 0.78);
                background-size: 28px 28px;
                border: 1px solid rgba(255, 255, 255, 0.22);
            }}

            .ai-band span {{
                display: inline-block;
                border-radius: 999px;
                padding: 5px 10px;
                background: rgba(180, 255, 213, 0.18);
                color: #dfffee !important;
                font-weight: 700;
                font-size: 0.78rem;
            }}

            .ai-band p {{
                color: #ffffff !important;
                font-weight: 700;
                text-shadow: 0 2px 10px rgba(0, 0, 0, 0.64);
            }}

            .auth-card {{
                border-radius: 8px;
                padding: 30px;
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid rgba(31, 122, 77, 0.14);
                box-shadow: 0 24px 64px rgba(15, 61, 43, 0.17);
            }}

            .auth-card h2,
            .auth-card h3 {{
                color: #0d6b43;
            }}

            .auth-card p,
            .auth-card span,
            .auth-card label {{
                color: #17382b;
            }}

            [data-testid="stTextInput"] input {{
                background: #fbfffc;
                color: #10281e !important;
                border: 1.5px solid rgba(31, 122, 77, 0.42);
                border-radius: 8px;
            }}

            [data-testid="stNumberInput"] input {{
                background: #fbfffc;
                color: #10281e !important;
                border: 1.5px solid rgba(31, 122, 77, 0.42);
            }}

            [data-testid="stNumberInput"] button {{
                background: #e7f7ee;
                color: #0f3d2b !important;
                border-color: rgba(31, 122, 77, 0.24);
            }}

            [data-testid="stNumberInput"] button * {{
                color: #0f3d2b !important;
            }}

            [data-testid="stTextInput"] input:focus {{
                border-color: #1f7a4d;
                box-shadow: 0 0 0 3px rgba(31, 122, 77, 0.14);
            }}

            div[data-testid="stVerticalBlock"]:has(> div [data-testid="stForm"]) {{
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid rgba(31, 122, 77, 0.18);
                border-radius: 8px;
                padding: 22px;
                box-shadow: 0 18px 54px rgba(15, 61, 43, 0.14);
            }}

            .stButton > button,
            .stFormSubmitButton > button {{
                border-radius: 8px;
                border: 0;
                background: linear-gradient(135deg, var(--leaf), #2fa978);
                color: #ffffff !important;
                font-weight: 700;
            }}

            .stButton > button *,
            .stFormSubmitButton > button * {{
                color: #ffffff !important;
            }}

            .stButton > button:hover,
            .stFormSubmitButton > button:hover {{
                border: 0;
                color: #ffffff !important;
                filter: brightness(0.96);
            }}

            div[data-testid="stTabs"] button {{
                color: var(--leaf-dark) !important;
                font-weight: 650;
            }}

            div[data-testid="stTabs"] button * {{
                color: var(--leaf-dark) !important;
            }}

            @media (max-width: 900px) {{
                .signal-grid {{
                    grid-template-columns: 1fr;
                }}
                .brand-panel {{
                    min-height: 460px;
                }}
                .ai-band {{
                    width: auto;
                    margin-top: 24px;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_login_page():
    st.markdown('<div class="login-hero">', unsafe_allow_html=True)
    left, right = st.columns([1.18, 0.82], gap="large")

    with left:
        st.markdown(
            """
            <section class="brand-panel">
                <div class="brand-kicker"><span class="leaf-mark"></span><span>AI AGRI PREDICT</span></div>
                <h1>AgriPredict</h1>
                <p>
                    Agriculture intelligence for yield planning, soil and crop recognition,
                    fertilizer estimates, and fast farm guidance.
                </p>
                <div class="signal-grid">
                    <div class="signal-card"><strong>AI</strong><span>Crop disease detection</span></div>
                    <div class="signal-card"><strong>NPK</strong><span>Fertilizer recommendation</span></div>
                    <div class="signal-card"><strong>Yield</strong><span>Crop yield prediction</span></div>
                    <div class="signal-card"><strong>Voice</strong><span>Farmer advisory assistant</span></div>
                </div>
                <div class="trust-strip">
                    <span class="trust-pill">Crop recommendation</span>
                    <span class="trust-pill">Soil recognition</span>
                    <span class="trust-pill">Yield prediction</span>
                    <span class="trust-pill">Voice assistant</span>
                    <span class="trust-pill">Weather intelligence ready</span>
                </div>
                <div class="ai-band">
                    <span>AI FIELD INTELLIGENCE</span>
                    <p>Model-assisted decisions grounded in farmer-entered field data.</p>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Welcome Back")
        st.caption("Sign in to access AI-powered agricultural insights.")
        login_tab, signup_tab = st.tabs(["Login", "Create Account"])

        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

            if submitted:
                try:
                    user = authenticate_user(email, password)
                    if user:
                        st.session_state.authenticated_user = user
                        st.success("Login successful.")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                except Exception as exc:
                    st.error(f"MongoDB connection failed: {exc}")

        with signup_tab:
            with st.form("signup_form"):
                full_name = st.text_input("Name")
                new_email = st.text_input("Email")
                new_password = st.text_input("New password", type="password")
                confirm_password = st.text_input("Confirm password", type="password")
                created = st.form_submit_button("Create account", use_container_width=True)

            if created:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    try:
                        user = create_user(new_email, new_password, full_name)
                        st.session_state.authenticated_user = user
                        st.success("Account created.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_resource
def load_yield_model():
    if not YIELD_MODEL_PATH.exists():
        return None
    return joblib.load(YIELD_MODEL_PATH)


@st.cache_resource
def load_image_models():
    if tf is None:
        return None, None

    try:
        soil_model = tf.keras.models.load_model(SOIL_MODEL_PATH) if SOIL_MODEL_PATH.exists() else None
    except Exception:
        soil_model = None

    try:
        crop_model = tf.keras.models.load_model(CROP_MODEL_PATH) if CROP_MODEL_PATH.exists() else None
    except Exception:
        crop_model = None

    return soil_model, crop_model


def load_class_labels(classes_path, dataset_dir, fallback):
    if classes_path.exists():
        with classes_path.open(encoding="utf-8") as fp:
            class_map = json.load(fp)
        return [canonical_label(class_map[str(index)]) for index in range(len(class_map))]

    if dataset_dir.exists():
        return [canonical_label(path.name) for path in sorted(dataset_dir.iterdir()) if path.is_dir()]

    return list(fallback)


def speak_result(text, lang):
    lang_code = TTS_LANG_CODES.get(lang, "en")
    tts = gTTS(text=text, lang=lang_code)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp


def recognize_speech_once():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Speak now...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=7)
            text = recognizer.recognize_google(audio, language="en-IN")
            st.success(f"You said: {text}")
            return text
    except Exception as exc:
        st.error(f"Speech error: {exc}")
        return ""


def calculate_recommendation(land_size, crop_key, soil_key):
    base = FERTILIZER_DATA[crop_key][soil_key]
    n_total = base["N"] * land_size
    p_total = base["P"] * land_size
    k_total = base["K"] * land_size
    n_bags = math.ceil(n_total / (BAG_WEIGHT_KG * UREA_N_PERCENT))
    p_bags = math.ceil((p_total * 2.29) / (BAG_WEIGHT_KG * DAP_P_PERCENT))
    k_bags = math.ceil((k_total * 1.2) / (BAG_WEIGHT_KG * MOP_K_PERCENT))
    return {
        "N": n_total,
        "P": p_total,
        "K": k_total,
        "N_bags": n_bags,
        "P_bags": p_bags,
        "K_bags": k_bags,
        "cost": n_bags * MRP_UREA + p_bags * MRP_DAP + k_bags * MRP_MOP,
    }


def predict_image(model, img_file, labels):
    if image is None:
        raise RuntimeError(f"TensorFlow is not available: {TENSORFLOW_IMPORT_ERROR}")

    img = image.load_img(img_file, target_size=(128, 128))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr, verbose=0)
    idx = int(np.argmax(pred))
    label = labels[idx] if idx < len(labels) else f"Class {idx}"
    return label, float(pred[0][idx])


def build_chatbot_response(user_message):
    message = user_message.lower().strip()
    if not message:
        return "Ask me about crop yield, soil health, fertilizer, irrigation, pests, or crop selection."

    matched_topics = [
        topic
        for topic in AGRI_TOPICS.values()
        if any(keyword in message for keyword in topic["keywords"])
    ]

    if not matched_topics:
        return (
            "I can help with crop selection, fertilizer planning, soil health, "
            "irrigation, pest symptoms, and yield improvement. Share crop name, "
            "soil type, season, and the issue you are seeing for a specific answer."
        )

    response_parts = [topic["answer"] for topic in matched_topics[:2]]
    response_parts.append(
        "For a sharper recommendation, share crop name, soil type, land size, "
        "recent rainfall, and visible symptoms or current growth stage."
    )
    response_parts.append(CHATBOT_DISCLAIMER)
    return "\n\n".join(response_parts)


st.set_page_config(page_title="AgriPredict", layout="wide")
apply_custom_theme()

if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None

if not st.session_state.authenticated_user:
    render_login_page()
    st.stop()

lang = st.sidebar.selectbox("Language", list(LANGUAGES.keys()), format_func=lambda key: LANGUAGES[key])
voice_enabled = st.sidebar.checkbox("Enable voice output", True)
active_user = st.session_state.authenticated_user
display_name = active_user.get("full_name") or active_user.get("email") or active_user.get("username")
st.sidebar.success(f"Signed in as {display_name}")
if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.authenticated_user = None
    st.rerun()

yield_model = load_yield_model()
soil_model, crop_model = load_image_models()
soil_labels = load_class_labels(SOIL_CLASSES_PATH, SOIL_DATASET_DIR, SOIL_NAMES.keys())
crop_labels = load_class_labels(CROP_CLASSES_PATH, CROP_DATASET_DIR, CROP_NAMES.keys())
t = {"title": ui_text(lang, "title"), "instruction": ui_text(lang, "instruction")}

st.title(t["title"])
st.write(t["instruction"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        ui_text(lang, "tab_yield"),
        ui_text(lang, "tab_classifier"),
        ui_text(lang, "tab_fertilizer"),
        ui_text(lang, "tab_chatbot"),
        ui_text(lang, "tab_database"),
    ]
)

with tab1:
    if yield_model is None:
        st.error(f"Yield model missing: {YIELD_MODEL_PATH.name}. Run `python train_yield_model.py`.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.number_input(ui_text(lang, "temperature"), value=25.0)
            rainfall = st.number_input(ui_text(lang, "rainfall"), value=1000.0)
            irrigation = st.number_input(ui_text(lang, "irrigation"), value=200.0)
            fert = st.number_input(ui_text(lang, "fertilizer_kg"), value=100.0)
        with col2:
            n = st.number_input(ui_text(lang, "soil_nitrogen"), value=30.0)
            p = st.number_input(ui_text(lang, "soil_phosphorus"), value=15.0)
            k = st.number_input(ui_text(lang, "soil_potassium"), value=40.0)

        if st.button(ui_text(lang, "speak_input")):
            recognize_speech_once()

        if st.button(ui_text(lang, "predict_yield"), type="primary"):
            row = pd.DataFrame(
                [[temperature, rainfall, irrigation, n, p, k, fert]],
                columns=FEATURE_COLUMNS,
            )
            try:
                pred = yield_model.predict(row)
                result = f"Predicted yield: {float(pred[0]):.2f} t/ha"
                st.success(result)
                if voice_enabled:
                    st.audio(speak_result(result, lang), format="audio/mp3")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tab2:
    classifier_choice_display = st.radio(
        ui_text(lang, "classifier"),
        [ui_text(lang, "soil"), ui_text(lang, "crop")],
        horizontal=True,
    )
    classifier_choice = "Soil" if classifier_choice_display == ui_text(lang, "soil") else "Crop"
    if TENSORFLOW_IMPORT_ERROR is not None:
        st.warning(
            "TensorFlow is not available in this Python environment, so image "
            f"classification is disabled. Details: {TENSORFLOW_IMPORT_ERROR}"
        )

    file = st.file_uploader(ui_text(lang, "upload_image"), type=["jpg", "jpeg", "png"])

    if file:
        st.image(file, caption=ui_text(lang, "uploaded_image"), use_container_width=True)
        if classifier_choice == "Soil" and soil_model is not None:
            label, conf = predict_image(soil_model, file, soil_labels)
            result = f"{translate_soil(label, lang)} ({conf:.2f})"
            st.success(result)
            if voice_enabled:
                st.audio(speak_result(result, lang), format="audio/mp3")
        elif classifier_choice == "Crop" and crop_model is not None:
            label, conf = predict_image(crop_model, file, crop_labels)
            result = f"{translate_crop(label, lang)} ({conf:.2f})"
            st.success(result)
            if voice_enabled:
                st.audio(speak_result(result, lang), format="audio/mp3")
        else:
            model_name = SOIL_MODEL_PATH.name if classifier_choice == "Soil" else CROP_MODEL_PATH.name
            st.error(f"{classifier_choice} classifier model missing: {model_name}")

with tab3:
    land = st.number_input(ui_text(lang, "land_size"), min_value=0.1, value=1.0, step=0.1)
    crop_choice = st.selectbox("Crop", list(FERTILIZER_DATA.keys()))
    soil_choice = st.selectbox("Soil", list(FERTILIZER_DATA[crop_choice].keys()))

    if st.button(ui_text(lang, "calculate_fertilizer"), type="primary"):
        res = calculate_recommendation(land, crop_choice, soil_choice)
        st.success(
            f"N: {res['N']:.1f} kg, P: {res['P']:.1f} kg, K: {res['K']:.1f} kg. "
            f"Bags - Urea: {res['N_bags']}, DAP: {res['P_bags']}, MOP: {res['K_bags']}. "
            f"Estimated cost: Rs {res['cost']:.2f}"
        )

with tab4:
    st.subheader("Agriculture Assistant")
    st.caption("Ask about crop selection, fertilizer, irrigation, soil health, pests, or yield improvement.")

    if "agri_chat_messages" not in st.session_state:
        st.session_state.agri_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello. Tell me your crop, soil type, season, and farming issue, "
                    "and I will suggest practical next steps."
                ),
            }
        ]

    quick_cols = st.columns(len(QUICK_QUESTIONS))
    for col, question in zip(quick_cols, QUICK_QUESTIONS):
        if col.button(question, key=f"quick_{question}"):
            st.session_state.agri_chat_messages.append({"role": "user", "content": question})
            st.session_state.agri_chat_messages.append(
                {"role": "assistant", "content": build_chatbot_response(question)}
            )

    for message in st.session_state.agri_chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask an agriculture question")
    if prompt:
        st.session_state.agri_chat_messages.append({"role": "user", "content": prompt})
        st.session_state.agri_chat_messages.append(
            {"role": "assistant", "content": build_chatbot_response(prompt)}
        )
        st.rerun()

    last_message = st.session_state.agri_chat_messages[-1]
    if voice_enabled and last_message["role"] == "assistant":
        if st.button("Listen to last chatbot answer"):
            st.audio(speak_result(last_message["content"], lang), format="audio/mp3")

with tab5:
    st.subheader(ui_text(lang, "yield_database"))
    try:
        st.metric(ui_text(lang, "records"), count_yield_records())
        if st.button(ui_text(lang, "import_sample_data")):
            total_rows = import_sample_data(replace=True)
            st.success(f"Imported {total_rows} records from sample_data.csv.")
            st.rerun()
    except Exception as exc:
        st.error(f"Database error: {exc}")
