import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
from pathlib import Path

# Custom CSS for premium look
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #ffffff;
    }
    
    /* Complete right panel area */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    
    /* Main content area - right panel */
    [data-testid="stMainBlockContainer"] {
        background-color: #ffffff !important;
    }
    
    /* Title styling */
    .title-main {
        text-align: center;
        color: #003366;
        font-size: 2.5em;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 1.1em;
        margin-bottom: 30px;
        font-style: italic;
    }
    
    .sidebar-section-title{
        color: #ffffff !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.8px;
        margin-top: 10px;
        margin-bottom: 10px;
        opacity: 1 !important;
        text-shadow: 0 0 2px rgba(255,255,255,0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #003366;
        margin-bottom: 10px;
    }
    
    .metric-label {
        color: #666666;
        font-size: 0.9em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        color: #003366;
        font-size: 2em;
        font-weight: 700;
        margin-top: 8px;
    }
    
    /* Defect badge */
    .defect-badge {
        display: inline-block;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1em;
        text-align: center;
        margin: 10px 0;
    }
    
    .badge-stable {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    
    .badge-lof {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
    }
    
    .badge-keyhole {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    
    /* Recommendation box */
    .recommendation-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #003366;
        margin-top: 20px;
    }
    
    .recommendation-title {
        color: #003366;
        font-weight: 700;
        font-size: 1.1em;
        margin-bottom: 10px;
    }
    
    .recommendation-text {
        color: #333333;
        font-size: 0.95em;
        line-height: 1.6;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #003366;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 30px;
        font-weight: 600;
        width: 100%;
        font-size: 1em;
    }
    
    .stButton > button:hover {
        background-color: #004d99;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background-color: #28a745;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 30px;
        font-weight: 600;
        width: 100%;
        font-size: 1em;
    }
    
    .stDownloadButton > button:hover {
        background-color: #218838;
    }
    
    /* Section headers */
    .section-header {
        color: #003366;
        font-size: 1.4em;
        font-weight: 700;
        border-bottom: 3px solid #003366;
        padding-bottom: 10px;
        margin-top: 25px;
        margin-bottom: 20px;
    }
    
    /* Slider styling */
    .stSlider {
        padding: 10px 0;
    }
    
    /* Text visibility */
    .stMarkdown {
        color: #333333 !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Constants
MODEL_DIR = Path("models")
DEFECT_MAPPING = {
    0: "Lack of Fusion",
    1: "Stable / Conduction",
    2: "Keyhole"
}

DEFECT_COLORS = {
    0: "badge-lof",      # Orange
    1: "badge-stable",   # Green
    2: "badge-keyhole"   # Red
}

# Load models with caching
@st.cache_resource
def load_models():
    """Load all trained models and feature order."""
    try:
        models = {
            'MeltPoolWidth_um': joblib.load(MODEL_DIR / 'MeltPoolWidth_um.pkl'),
            'MeltPoolDepth_um': joblib.load(MODEL_DIR / 'MeltPoolDepth_um.pkl'),
            'defect_classifier': joblib.load(MODEL_DIR / 'defect_classifier.pkl'),
            'YieldStrength_MPa': joblib.load(MODEL_DIR / 'YieldStrength_MPa.pkl'),
            'UTS_MPa': joblib.load(MODEL_DIR / 'UTS_MPa.pkl'),
            'Hardness_HRA': joblib.load(MODEL_DIR / 'Hardness_HRA.pkl'),
            'Elongation_pct': joblib.load(MODEL_DIR / 'Elongation_pct.pkl'),
        }
        
        # Load feature order
        feature_order = joblib.load(MODEL_DIR / 'feature_order.pkl')
        
        return models, feature_order
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def predict_all(models, feature_order, input_data):
    """Run all 7 models and return predictions."""
    try:
        # Reorder features according to model training
        X = input_data[feature_order].copy()
        
        predictions = {
            'MeltPoolWidth_um': float(models['MeltPoolWidth_um'].predict(X)[0]),
            'MeltPoolDepth_um': float(models['MeltPoolDepth_um'].predict(X)[0]),
            'DefectClass': int(models['defect_classifier'].predict(X)[0]),
            'YieldStrength_MPa': float(models['YieldStrength_MPa'].predict(X)[0]),
            'UTS_MPa': float(models['UTS_MPa'].predict(X)[0]),
            'Hardness_HRA': float(models['Hardness_HRA'].predict(X)[0]),
            'Elongation_pct': float(models['Elongation_pct'].predict(X)[0]),
        }
        
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def calculate_quality_score(predictions):
    """
    Calculate dynamic continuous engineering quality score (0-100).
    Uses smooth penalty functions for material properties optimization.
    
    Scoring breakdown:
    - Defect classification: 40 points (most critical)
    - Meltpool geometry: 25 points (Gaussian penalty around optimal)
    - Strength properties: 25 points (linear ramp to targets)
    - Ductility: 10 points (Gaussian penalty around 18%)
    
    Total: 100 points
    """
    score = 0
    
    # ---------------------------------
    # 1. DEFECT SCORE (40 points)
    # ---------------------------------
    defect = predictions["DefectClass"]
    
    if defect == 1:  # Stable / Conduction
        score += 40
    elif defect == 0:  # Lack of Fusion
        score += 24
    else:  # Keyhole
        score += 18
    
    # ---------------------------------
    # 2. MELTPOOL GEOMETRY SCORE (25 points)
    # ---------------------------------
    # Gaussian penalty around optimal values
    width = predictions["MeltPoolWidth_um"]
    depth = predictions["MeltPoolDepth_um"]
    
    # Width score: optimal at 160 µm, decays with deviation
    width_score = max(0, 12.5 - abs(width - 160) / 8)
    
    # Depth score: optimal at 95 µm, decays with deviation
    depth_score = max(0, 12.5 - abs(depth - 95) / 6)
    
    score += width_score + depth_score
    
    # ---------------------------------
    # 3. STRENGTH SCORE (25 points)
    # ---------------------------------
    # Linear ramp scoring towards target values
    ys = predictions["YieldStrength_MPa"]
    uts = predictions["UTS_MPa"]
    
    # Yield strength: 10 points max, target ~755 MPa
    # Scales from 700 MPa (0 points) to 1000+ MPa (10 points)
    ys_score = max(0, min(10, (ys - 700) / 12))
    
    # UTS: 15 points max, target ~1010 MPa
    # Scales from 850 MPa (0 points) to 1030+ MPa (15 points)
    uts_score = max(0, min(15, (uts - 850) / 12))
    
    score += ys_score + uts_score
    
    # ---------------------------------
    # 4. DUCTILITY SCORE (10 points)
    # ---------------------------------
    # Gaussian penalty around optimal 18% elongation
    elong = predictions["Elongation_pct"]
    
    # 10 points at 18%, decays with deviation (coefficient 1.2)
    elong_score = max(0, 10 - abs(elong - 18) * 1.2)
    
    score += elong_score
    score = round(score)
    # Return final score (0-100)
    return round(min(100, score), 1)

def get_recommendation(defect_class):
    """Get recommendation based on defect class."""
    if defect_class == 0:  # Lack of Fusion
        return "⚠️ **Lack of Fusion Detected**\n\nTo improve print quality:\n- **Increase Power** (W) to enhance melt pool penetration\n- **Reduce Scan Speed** (mm/s) to allow longer dwell time"
    elif defect_class == 2:  # Keyhole
        return "⚠️ **Keyhole Defect Detected**\n\nTo improve print quality:\n- **Reduce Power** (W) to prevent excessive vaporization\n- **Increase Scan Speed** (mm/s) to reduce dwell time"
    else:  # Stable
        return "✅ **Stable Processing Conditions**\n\nYour parameters are within the optimal process window. This is an excellent region for consistent, high-quality prints with:\n- Good melt pool geometry\n- Minimal defects\n- Predictable mechanical properties"

def show_metric_card(label, value, unit=""):
    """Display a metric card."""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value:.2f} {unit}</div>
        </div>
    """, unsafe_allow_html=True)

def create_defect_badge(defect_class):
    """Create colored defect status badge."""
    defect_name = DEFECT_MAPPING.get(defect_class, "Unknown")
    badge_class = DEFECT_COLORS.get(defect_class, "badge-stable")
    
    st.markdown(f"""
        <div class="defect-badge {badge_class}">
            {defect_name}
        </div>
    """, unsafe_allow_html=True)

def export_to_csv(input_params, predictions):
    """Create CSV export data."""
    export_data = {
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'Power_W': [input_params['P_W']],
        'ScanSpeed_mm_per_s': [input_params['v_mm_per_s']],
        'HatchSpacing_mm': [input_params['h_mm']],
        'LayerThickness_mm': [input_params['t_mm']],
        'MeltPoolWidth_um': [predictions['MeltPoolWidth_um']],
        'MeltPoolDepth_um': [predictions['MeltPoolDepth_um']],
        'DefectClass': [DEFECT_MAPPING[predictions['DefectClass']]],
        'YieldStrength_MPa': [predictions['YieldStrength_MPa']],
        'UTS_MPa': [predictions['UTS_MPa']],
        'Hardness_HRA': [predictions['Hardness_HRA']],
        'Elongation_pct': [predictions['Elongation_pct']],
    }
    
    df = pd.DataFrame(export_data)
    return df.to_csv(index=False).encode('utf-8')

# Main app
def main():
    # Header
    st.markdown('<div class="title-main">🔬 HAYNES 282 DIGITAL TWIN</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">LPBF Process Parameter Intelligence Platform</div>', unsafe_allow_html=True)
    
    # Load models
    models, feature_order = load_models()
    
    if models is None or feature_order is None:
        st.error("Failed to load models. Please check the Models folder.")
        return
    
    # Sidebar: Input parameters
    st.sidebar.markdown(
        """
        <div class="sidebar-section-title">
            ⚙️ PROCESS PARAMETERS
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")
    
    power_w = st.sidebar.slider(
        "Power (W)",
        min_value=150,
        max_value=370,
        value=250,
        step=1,
        help="Laser power in watts"
    )
    
    scan_speed = st.sidebar.slider(
        "Scan Speed (mm/s)",
        min_value=350,
        max_value=1772,
        value=1000,
        step=10,
        help="Laser scan speed in mm/s"
    )
    
    hatch_spacing = st.sidebar.slider(
        "Hatch Spacing (mm)",
        min_value=0.09,
        max_value=0.12,
        value=0.10,
        step=0.01,
        format="%.2f",
        help="Distance between hatch lines in mm"
    )
    
    layer_thickness = st.sidebar.slider(
        "Layer Thickness (mm)",
        min_value=0.03,
        max_value=0.05,
        value=0.04,
        step=0.01,
        format="%.2f",
        help="Thickness of each powder layer in mm"
    )
    
    st.sidebar.markdown("---")
    
    # Predict button
    predict_clicked = st.sidebar.button("🔮 PREDICT", use_container_width=True)
    
    # Prepare input data with exact training feature names
    input_params = {
        'P_W': power_w,
        'v_mm_per_s': scan_speed,
        'h_mm': hatch_spacing,
        't_mm': layer_thickness
    }
    
    input_df = pd.DataFrame([input_params])
    
    # Run predictions
    if predict_clicked:
        with st.spinner("🔄 Running predictions..."):
            predictions = predict_all(models, feature_order, input_df)
        
        if predictions is not None:
            # Store in session state for export
            st.session_state.forward_predictions = predictions
            st.session_state.forward_input_params = input_params
            st.session_state.forward_quality_score = calculate_quality_score(predictions)
            st.session_state.forward_show_results = True
    
    # Display results
    if st.session_state.get('forward_show_results', False):
        predictions = st.session_state.forward_predictions
        quality_score = st.session_state.forward_quality_score
        
        # Quality score gauge
        st.markdown('<div class="section-header">📊 PROCESS QUALITY ASSESSMENT</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=quality_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Quality Score"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#003366"},
                    'steps': [
                        {'range': [0, 50], 'color': "#f8d7da"},
                        {'range': [50, 80], 'color': "#fff3cd"},
                        {'range': [80, 100], 'color': "#d4edda"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("###")
            col2a, col2b = st.columns(2)
            with col2a:
                show_metric_card("Score", quality_score, "")
            with col2b:
                show_metric_card("Process", predictions['DefectClass'], "")
        
        # Meltpool Geometry
        st.markdown('<div class="section-header">🌡️ MELTPOOL GEOMETRY</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            show_metric_card("Melt Pool Width", predictions['MeltPoolWidth_um'], "µm")
        with col2:
            show_metric_card("Melt Pool Depth", predictions['MeltPoolDepth_um'], "µm")
        
        # Defect Status
        st.markdown('<div class="section-header">⚠️ DEFECT STATUS</div>', unsafe_allow_html=True)
        create_defect_badge(predictions['DefectClass'])
        
        # Mechanical Properties
        st.markdown('<div class="section-header">💪 MECHANICAL PROPERTIES</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_metric_card("Yield Strength", predictions['YieldStrength_MPa'], "MPa")
        with col2:
            show_metric_card("UTS", predictions['UTS_MPa'], "MPa")
        with col3:
            show_metric_card("Hardness", predictions['Hardness_HRA'], "HRA")
        with col4:
            show_metric_card("Elongation", predictions['Elongation_pct'], "%")
        
        # # Mechanical Properties Comparison Chart
        # st.markdown('<div class="section-header">📈 PROPERTIES COMPARISON</div>', unsafe_allow_html=True)
        
        # # Normalize values for better visualization
        # props_data = {
        #     'Property': ['Yield Strength\n(MPa)', 'UTS\n(MPa)', 'Hardness\n(HRA)', 'Elongation\n(%)'],
        #     'Value': [
        #         predictions['YieldStrength_MPa'] / 10,  # Scale for visibility
        #         predictions['UTS_MPa'] / 10,
        #         predictions['Hardness_HRA'],
        #         predictions['Elongation_pct'] * 10
        #     ]
        # }
        
        # fig = go.Figure(data=[
        #     go.Bar(
        #         x=props_data['Property'],
        #         y=props_data['Value'],
        #         marker=dict(color=['#003366', '#004d99', '#336699', '#6699cc']),
        #         text=[f"{predictions['YieldStrength_MPa']:.1f}", 
        #               f"{predictions['UTS_MPa']:.1f}",
        #               f"{predictions['Hardness_HRA']:.1f}",
        #               f"{predictions['Elongation_pct']:.1f}"],
        #         textposition='outside'
        #     )
        # ])
        # fig.update_layout(height=400, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        # st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="recommendation-title">💡 RECOMMENDATION</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="recommendation-text">{get_recommendation(predictions["DefectClass"])}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export section
        st.markdown('<div class="section-header">📥 EXPORT RESULTS</div>', unsafe_allow_html=True)
        
        csv_data = export_to_csv(st.session_state.forward_input_params, predictions)
        st.download_button(
            label="📥 Download Prediction Report (CSV)",
            data=csv_data,
            file_name=f"HAYNES282_Prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Input parameters summary
        st.markdown('<div class="section-header">📋 INPUT PARAMETERS USED</div>', unsafe_allow_html=True)
        
        # Map technical names to user-friendly labels
        params_display = {
            'Power (W)': st.session_state.forward_input_params['P_W'],
            'Scan Speed (mm/s)': st.session_state.forward_input_params['v_mm_per_s'],
            'Hatch Spacing (mm)': st.session_state.forward_input_params['h_mm'],
            'Layer Thickness (mm)': st.session_state.forward_input_params['t_mm'],
        }
        params_df = pd.DataFrame([params_display]).T
        params_df.columns = ['Value']
        st.dataframe(params_df, use_container_width=True)
    
    else:
        # Initial state message
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                ### 👋 Welcome to HAYNES 282 Digital Twin
                
                This platform uses advanced machine learning models trained on LPBF experimental data to predict:
                
                **Geometry:**
                - Melt Pool Width & Depth
                
                **Quality:**
                - Defect Classification
                
                **Mechanical Properties:**
                - Yield Strength, UTS, Hardness, Elongation
                
                #### How to use:
                1. Adjust process parameters in the sidebar (Power, Scan Speed, Hatch Spacing, Layer Thickness)
                2. Click the **PREDICT** button
                3. View detailed predictions and recommendations
                4. Export results as CSV
                
                ---
                
                **Start by adjusting the parameters in the sidebar and clicking PREDICT!**
            """)

if __name__ == "__main__":
    # Initialize session state
    if 'forward_show_results' not in st.session_state:
        st.session_state.forward_show_results = False
    if 'forward_predictions' not in st.session_state:
        st.session_state.forward_predictions = None
    if 'forward_input_params' not in st.session_state:
        st.session_state.forward_input_params = None
    if 'forward_quality_score' not in st.session_state:
        st.session_state.forward_quality_score = None
    
    main()
