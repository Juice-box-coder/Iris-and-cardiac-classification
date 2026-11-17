"""
Assignment 1: Machine Learning Classification Systems
Pentecost University - Faculty of Computing and Information Systems (FOCIS)
Supervised by: Mr. Harry Attieku-Boateng
Student: Emmanuel Asante (PU092301 / PUIT23210016)

Full-Screen Iris Classification & Biometrics Inference Deck
Directly renders the single-file web application without any tab widgets.
"""

import os
import streamlit as st
import streamlit.components.v1 as components

# Configure Full-Width Page
st.set_page_config(
    page_title="Iris Classification & Biometrics Deck",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Strip away all Streamlit container paddings, headers, and footers
st.markdown("""
<style>
    /* Remove default Streamlit chrome */
    #MainMenu, header, footer { visibility: hidden !important; display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    iframe {
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        display: block !important;
    }
</style>
""", unsafe_allow_html=True)

# Load and render the full Mossline Web Application (v2.1 - Inlined Base64 Assets)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

if os.path.exists(INDEX_PATH):
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
    # Embed the entire web application into the full browser viewport
    components.html(html_content, height=12000, scrolling=True)
else:
    st.error("index.html not found.")
