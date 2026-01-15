import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Eco-Bags Delivery", 
    page_icon="🗺️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        .block-container { padding: 0.5rem 1rem 0rem 1rem; }
        header, footer { visibility: hidden; height: 0rem; }
        h3 { margin: 0 0 0.5rem 0 !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 🗺️ Eco-Bags Delivery Optimizer")
st.caption("Locations visualized, ready for prioritization and route planning.")

# Load and display the HTML map
script_dir = Path(__file__).resolve().parent
map_path = script_dir.parent / "maps" / "showcase_delivery_locations.html"

with open(map_path, "r", encoding="utf-8") as f:
    st.components.v1.html(f.read(), height=800, scrolling=False)