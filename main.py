import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Social Media Engagement Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### Predictive Analytics Project\nSocial Media Engagement Prediction"
    }
)

from processor import DataProcessor
from ui import render_multi_tab_ui as render_ui
import warnings
warnings.filterwarnings('ignore')

# Dark theme CSS injection
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #262730; }
    .stTabs [data-baseweb="tab"] { background-color: #262730; color: white; }
    .stTabs [aria-selected="true"] { background-color: #0e1117; border-color: #4CAF50; }
    .st-bb, .st-at { background-color: #1a1a1a; }
    .css-1d391kg { background-color: #333333; }
    .stPlotlyChart { border: 1px solid #4CAF50; border-radius: 5px; }
    .stDataFrame { background-color: #262730 !important; }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize processor with dataset if not already in session state
    if 'processor' not in st.session_state:
        # Use file open dialog instead of hardcoded path
        uploaded_file = None
        
        # During initial load, no file will be uploaded yet
        # Initialize processor with empty dataframe for now
        st.session_state.processor = DataProcessor(uploaded_file)
        st.session_state.file_uploaded = False
        
    # Render UI with the processor
    render_ui(st.session_state.processor)

if __name__ == "__main__":
    main()
