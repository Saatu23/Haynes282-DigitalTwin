import streamlit as st

# Configure page - MUST be first Streamlit command  
st.set_page_config(
    page_title="HAYNES 282 Digital Twin",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for all pages
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'input_params' not in st.session_state:
    st.session_state.input_params = None
if 'quality_score' not in st.session_state:
    st.session_state.quality_score = None
if 'inverse_results' not in st.session_state:
    st.session_state.inverse_results = None

# Premium CSS Design
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #003366 0%, #004d99 100%);
    }
    
    .title-main {
        text-align: center;
        background: linear-gradient(135deg, #003366 0%, #004d99 100%);
        color: white;
        font-size: 2.8em;
        font-weight: 800;
        letter-spacing: 2px;
        margin: -1.5rem -1.5rem 0 -1.5rem;
        padding: 3rem 1.5rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 10px 30px rgba(0, 51, 102, 0.2);
    }
    
    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 1.2em;
        margin-bottom: 3rem;
        font-style: italic;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    .welcome-content {
        background: white;
        border-radius: 15px;
        padding: 3rem;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        color: #333333;
        font-size: 1.05em;
        line-height: 1.9;
    }
    
    .welcome-content h3 {
        color: #003366;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        font-size: 1.5em;
        border-left: 5px solid #004d99;
        padding-left: 1rem;
    }
    
    .welcome-content strong {
        color: #004d99;
        font-weight: 700;
    }
    
    .welcome-content p {
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Home page content
st.markdown('<div class="title-main">HAYNES 282 DIGITAL TWIN</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">LPBF Process Parameter Intelligence Platform</div>', unsafe_allow_html=True)

st.markdown("""
<div class="welcome-content">

### Welcome to HAYNES 282 Digital Twin

This comprehensive platform uses advanced machine learning models trained on LPBF experimental data to optimize your additive manufacturing process.

**What you can do:**

**• Forward Prediction** - Predict material properties and defects from process parameters
- Input: Power, Scan Speed, Hatch Spacing, Layer Thickness
- Output: Meltpool geometry, defect classification, mechanical properties
- Get: Quality score and process recommendations

**• Inverse Optimizer** - Find optimal process parameters for your target properties
- Input: Target mechanical properties and constraints
- Output: Recommended Power, Scan Speed, Hatch Spacing, Layer Thickness
- Get: Verified parameter set with predicted performance

---

**Select a tab from the sidebar to get started!**

</div>
""", unsafe_allow_html=True)
