import streamlit as st
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import numpy as np

from PIL import Image
from ultralytics import YOLO

import pandas as pd
import base64
from io import BytesIO
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# -------------------------------------------------
# 1. SETUP PATHS
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "brain_tumor_seg4",
    "weights",
    "best.pt"
)

# -------------------------------------------------
# 2. LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_yolo():
    if os.path.exists(MODEL_PATH):
        try:
            return YOLO(MODEL_PATH), True
        except:
            return None, False
    return None, False

model, is_active = load_yolo()

# -------------------------------------------------
# 2B. HELPER FUNCTIONS
# -------------------------------------------------
def get_image_download_link(img_array, filename="image.png"):
    """Generate download link for processed images"""
    pil_img = Image.fromarray(img_array)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/png;base64,{img_str}" download="{filename}" style="text-decoration: none; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.6rem 1.2rem; border-radius: 8px; font-weight: 600; display: inline-block; margin-top: 0.5rem;">💾 Download Image</a>'

def calculate_tumor_volume(area_px, pixel_spacing=0.5):
    """Estimate tumor volume (simplified calculation)"""
    area_mm = area_px * (pixel_spacing ** 2)
    estimated_volume_mm3 = area_mm * 5  # Assuming 5mm slice thickness
    return area_mm, estimated_volume_mm3

def get_clinical_recommendation(stage, area, tumor_type):
    """Generate clinical recommendations based on findings"""
    recommendations = {
        "Low Risk": [
            "Continue regular monitoring with periodic MRI scans",
            "Clinical review recommended in 3-6 months",
            "Document baseline measurements for comparison"
        ],
        "Moderate Risk": [
            "Immediate consultation with neurosurgeon recommended",
            "Consider advanced imaging (MRI with contrast, PET scan)",
            "Multidisciplinary team review advised",
            "Follow-up scan in 1-3 months"
        ],
        "High Risk": [
            "Urgent neurosurgical consultation required",
            "Comprehensive pre-operative assessment needed",
            "Consider biopsy for definitive diagnosis",
            "Immediate treatment planning recommended"
        ]
    }
    
    if area < 5000:
        return recommendations["Low Risk"]
    elif area < 15000:
        return recommendations["Moderate Risk"]
    else:
        return recommendations["High Risk"]

# -------------------------------------------------
# 3. PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI - Advanced Medical Imaging",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# 4. CUSTOM CSS STYLING
# -------------------------------------------------
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Section */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.3;
    }
    
    .header-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        color: white;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Professional Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 4px solid;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card.success {
        border-left-color: #10b981;
    }
    
    .metric-card.warning {
        border-left-color: #f59e0b;
    }
    
    .metric-card.danger {
        border-left-color: #ef4444;
    }
    
    .metric-card.info {
        border-left-color: #3b82f6;
    }
    
    .tumor-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .stage-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.3px;
    }
    
    .metric-item {
        display: flex;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 1px solid #f3f4f6;
    }
    
    .metric-item:last-child {
        border-bottom: none;
    }
    
    .metric-label {
        color: #6b7280;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .metric-value {
        color: #1f2937;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 10px 10px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Image Container */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        background: white;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Upload Section */
    .upload-section {
        background: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        border: 2px dashed #d1d5db;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #667eea;
        background: #f9fafb;
    }
    
    /* Statistics Dashboard */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Recommendation Box */
    .recommendation-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-top: 1rem;
    }
    
    .recommendation-box h4 {
        margin: 0 0 1rem 0;
        font-size: 1.2rem;
    }
    
    .recommendation-box ul {
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .recommendation-box li {
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Download Button Styling */
    .download-section {
        margin-top: 1rem;
        text-align: center;
    }
    
    /* Patient Info Box */
    .patient-info-box {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Comparison Table */
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .comparison-table th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        text-align: left;
        font-weight: 600;
    }
    
    .comparison-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .comparison-table tr:hover {
        background: #f9fafb;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: #6b7280;
        font-size: 0.9rem;
        border-top: 1px solid #e5e7eb;
    }
    
    /* Status Indicators */
    .status-active {
        color: #10b981;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .status-inactive {
        color: #ef4444;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Progress Bar Custom Styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Clean Scan Result */
    .clean-scan {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: #166534;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(150, 230, 161, 0.3);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# 5. PROFESSIONAL HEADER
# -------------------------------------------------
# Note: Header will be rendered within each page function

# -------------------------------------------------
# 6. PAGE NAVIGATION & SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Select Page",
        ["🔬 Analysis", "📊 Metrics & Performance"],
        label_visibility="collapsed"
    )
    st.markdown("---")

# -------------------------------------------------
# 7. METRICS PAGE FUNCTION
# -------------------------------------------------
def render_metrics_page():
    """Render the metrics and performance analysis page"""
    
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">📊 Model Performance Metrics</h1>
        <p class="header-subtitle">Comprehensive AI Model Analytics & Statistics</p>
        <span class="header-badge">✨ Real-time Performance Monitoring</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if results file exists
    results_path = os.path.join(PROJECT_ROOT, "results", "brain_tumor_seg4", "results.csv")
    
    if os.path.exists(results_path):
        # Load training results
        df_results = pd.read_csv(results_path)
        df_results.columns = df_results.columns.str.strip()
        
        # Overview Metrics
        st.markdown('<p class="section-header">🎯 Model Overview</p>', unsafe_allow_html=True)
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            epochs = len(df_results)
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{epochs}</div>
                <div class="stat-label">Training Epochs</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            if 'metrics/mAP50(B)' in df_results.columns:
                final_map50 = df_results['metrics/mAP50(B)'].iloc[-1]
            else:
                final_map50 = 0
            st.markdown(f"""
            <div class="stat-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="stat-number">{final_map50:.3f}</div>
                <div class="stat-label">mAP@50 (Final)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            if 'metrics/precision(B)' in df_results.columns:
                final_precision = df_results['metrics/precision(B)'].iloc[-1]
            else:
                final_precision = 0
            st.markdown(f"""
            <div class="stat-box" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                <div class="stat-number">{final_precision:.3f}</div>
                <div class="stat-label">Precision</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col4:
            if 'metrics/recall(B)' in df_results.columns:
                final_recall = df_results['metrics/recall(B)'].iloc[-1]
            else:
                final_recall = 0
            st.markdown(f"""
            <div class="stat-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="stat-number">{final_recall:.3f}</div>
                <div class="stat-label">Recall</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Training Performance Charts
        st.markdown('<p class="section-header">📈 Training Performance</p>', unsafe_allow_html=True)
        
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📉 Loss Curves", "🎯 Metrics Evolution", "🔍 Detailed Analysis"])
        
        with chart_tab1:
            # Loss curves
            fig_loss = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Box Loss', 'Segmentation Loss')
            )
            
            # Box Loss
            if 'train/box_loss' in df_results.columns:
                fig_loss.add_trace(
                    go.Scatter(x=df_results.index, y=df_results['train/box_loss'],
                              name='Train Box Loss', line=dict(color='#667eea', width=2)),
                    row=1, col=1
                )
            if 'val/box_loss' in df_results.columns:
                fig_loss.add_trace(
                    go.Scatter(x=df_results.index, y=df_results['val/box_loss'],
                              name='Val Box Loss', line=dict(color='#f5576c', width=2)),
                    row=1, col=1
                )
            
            # Segmentation Loss
            if 'train/seg_loss' in df_results.columns:
                fig_loss.add_trace(
                    go.Scatter(x=df_results.index, y=df_results['train/seg_loss'],
                              name='Train Seg Loss', line=dict(color='#667eea', width=2)),
                    row=1, col=2
                )
            if 'val/seg_loss' in df_results.columns:
                fig_loss.add_trace(
                    go.Scatter(x=df_results.index, y=df_results['val/seg_loss'],
                              name='Val Seg Loss', line=dict(color='#f5576c', width=2)),
                    row=1, col=2
                )
            
            fig_loss.update_layout(
                height=400,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            fig_loss.update_xaxes(title_text="Epoch", row=1, col=1)
            fig_loss.update_xaxes(title_text="Epoch", row=1, col=2)
            fig_loss.update_yaxes(title_text="Loss", row=1, col=1)
            fig_loss.update_yaxes(title_text="Loss", row=1, col=2)
            
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with chart_tab2:
            # Metrics evolution
            fig_metrics = go.Figure()
            
            if 'metrics/precision(B)' in df_results.columns:
                fig_metrics.add_trace(go.Scatter(
                    x=df_results.index, 
                    y=df_results['metrics/precision(B)'],
                    name='Precision',
                    line=dict(color='#10b981', width=2)
                ))
            
            if 'metrics/recall(B)' in df_results.columns:
                fig_metrics.add_trace(go.Scatter(
                    x=df_results.index, 
                    y=df_results['metrics/recall(B)'],
                    name='Recall',
                    line=dict(color='#f59e0b', width=2)
                ))
            
            if 'metrics/mAP50(B)' in df_results.columns:
                fig_metrics.add_trace(go.Scatter(
                    x=df_results.index, 
                    y=df_results['metrics/mAP50(B)'],
                    name='mAP@50',
                    line=dict(color='#667eea', width=3)
                ))
            
            if 'metrics/mAP50-95(B)' in df_results.columns:
                fig_metrics.add_trace(go.Scatter(
                    x=df_results.index, 
                    y=df_results['metrics/mAP50-95(B)'],
                    name='mAP@50-95',
                    line=dict(color='#ef4444', width=2)
                ))
            
            fig_metrics.update_layout(
                title="Detection Metrics Over Training",
                xaxis_title="Epoch",
                yaxis_title="Score",
                height=450,
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with chart_tab3:
            # Detailed metrics table
            st.markdown("##### 📋 Metrics by Epoch")
            
            # Select relevant columns
            display_cols = []
            for col in ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 
                       'metrics/mAP50-95(B)', 'train/box_loss', 'val/box_loss']:
                if col in df_results.columns:
                    display_cols.append(col)
            
            if display_cols:
                display_df = df_results[display_cols].copy()
                display_df.index.name = 'Epoch'
                display_df = display_df.round(4)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No detailed metrics available")
        
        st.markdown("---")
        
        # Model Configuration
        st.markdown('<p class="section-header">⚙️ Model Configuration</p>', unsafe_allow_html=True)
        
        args_path = os.path.join(PROJECT_ROOT, "results", "brain_tumor_seg4", "args.yaml")
        if os.path.exists(args_path):
            import yaml
            try:
                with open(args_path, 'r') as f:
                    args = yaml.safe_load(f)
                
                config_col1, config_col2, config_col3 = st.columns(3)
                
                with config_col1:
                    st.markdown("""
                    <div class="metric-card info">
                        <h4 style='color: #3b82f6;'>📐 Model Settings</h4>
                    """, unsafe_allow_html=True)
                    if isinstance(args, dict):
                        for key in ['model', 'imgsz', 'batch']:
                            if key in args:
                                st.write(f"**{key}:** {args[key]}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with config_col2:
                    st.markdown("""
                    <div class="metric-card success">
                        <h4 style='color: #10b981;'>🎯 Training Params</h4>
                    """, unsafe_allow_html=True)
                    if isinstance(args, dict):
                        for key in ['epochs', 'lr0', 'optimizer']:
                            if key in args:
                                st.write(f"**{key}:** {args[key]}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with config_col3:
                    st.markdown("""
                    <div class="metric-card warning">
                        <h4 style='color: #f59e0b;'>🔧 Augmentation</h4>
                    """, unsafe_allow_html=True)
                    if isinstance(args, dict):
                        for key in ['hsv_h', 'hsv_s', 'degrees']:
                            if key in args:
                                st.write(f"**{key}:** {args[key]}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            except Exception as e:
                st.warning(f"Could not load configuration: {e}")
        
        st.markdown("---")
        
        # Performance Summary
        st.markdown('<p class="section-header">📊 Performance Summary</p>', unsafe_allow_html=True)
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown("##### 🎯 Best Performance Indicators")
            
            best_metrics = {}
            if 'metrics/mAP50(B)' in df_results.columns:
                best_metrics['Best mAP@50'] = f"{df_results['metrics/mAP50(B)'].max():.4f}"
            if 'metrics/precision(B)' in df_results.columns:
                best_metrics['Best Precision'] = f"{df_results['metrics/precision(B)'].max():.4f}"
            if 'metrics/recall(B)' in df_results.columns:
                best_metrics['Best Recall'] = f"{df_results['metrics/recall(B)'].max():.4f}"
            
            for metric, value in best_metrics.items():
                st.metric(label=metric, value=value)
        
        with summary_col2:
            st.markdown("##### 📉 Loss Convergence")
            
            if 'train/box_loss' in df_results.columns:
                initial_loss = df_results['train/box_loss'].iloc[0]
                final_loss = df_results['train/box_loss'].iloc[-1]
                improvement = ((initial_loss - final_loss) / initial_loss * 100)
                
                st.metric(
                    label="Box Loss Reduction",
                    value=f"{final_loss:.4f}",
                    delta=f"-{improvement:.1f}%"
                )
            
            if 'train/seg_loss' in df_results.columns:
                initial_loss = df_results['train/seg_loss'].iloc[0]
                final_loss = df_results['train/seg_loss'].iloc[-1]
                improvement = ((initial_loss - final_loss) / initial_loss * 100)
                
                st.metric(
                    label="Seg Loss Reduction",
                    value=f"{final_loss:.4f}",
                    delta=f"-{improvement:.1f}%"
                )
        
    else:
        st.warning("⚠️ Training results not found. Please train the model first.", icon="📁")
        st.info(f"Expected path: `{results_path}`")
        
        # Show sample metrics structure
        st.markdown("---")
        st.markdown("### 📝 Sample Metrics Structure")
        st.code("""
        The metrics page will display:
        - Training/Validation Loss curves
        - mAP, Precision, Recall evolution
        - Model configuration details
        - Performance summaries
        
        Train your model to see real metrics here!
        """)

# -------------------------------------------------
# 8. ANALYSIS PAGE FUNCTION
# -------------------------------------------------
def render_analysis_page():
    """Render the main tumor analysis page"""
    
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">🧠 NeuroScan AI Platform</h1>
        <p class="header-subtitle">Advanced Brain Tumor Detection & Segmentation System</p>
        <span class="header-badge">✨ Powered by YOLOv8 Deep Learning | Clinical Grade Analysis</span>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Configuration Panel")
        st.markdown("---")

        # Model Status
        st.markdown("#### 🔬 System Status")
        if is_active:
            st.markdown('<div class="status-active">● AI Engine Active</div>', unsafe_allow_html=True)
            st.success("Model loaded successfully", icon="✅")
        else:
            st.markdown('<div class="status-inactive">● AI Engine Offline</div>', unsafe_allow_html=True)
            st.error("Model not available", icon="❌")

        st.markdown("---")

        # Detection Parameters
        st.markdown("#### 🎯 Detection Settings")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Minimum confidence score for tumor detection",
            key="conf_slider_analysis"
        )

        border_thickness = st.slider(
            "Border Thickness",
            min_value=1,
            max_value=5,
            value=2,
            help="Thickness of segmentation boundary overlay",
            key="border_slider_analysis"
        )

        st.markdown("---")

        # Model Information
        if is_active:
            st.markdown("#### 🧬 Tumor Classes")
            for i, name in model.names.items():
                st.markdown(f"""
                <div style='background: rgba(102, 126, 234, 0.1); 
                            padding: 0.5rem; 
                            border-radius: 8px; 
                            margin: 0.3rem 0;
                            border-left: 3px solid #667eea;'>
                    <b>Class {i}:</b> {name.upper()}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Additional Info
        st.markdown("#### ℹ️ About")
        st.info("This platform uses state-of-the-art AI to analyze MRI scans and detect brain tumors with high precision.", icon="💡")
        
        # Quick Tips
        with st.expander("💡 Quick Tips"):
            st.markdown("""
            **📤 Upload:**
            - Supports multiple scans at once
            - Formats: JPG, PNG, JPEG
            
            **🎯 Settings:**
            - Higher confidence = fewer false positives
            - Adjust border thickness for clarity
            
            **📊 Features:**
            - View detection, segmentation & heatmaps
            - Download all result images
            - Compare multiple scans
            - Clinical recommendations provided
            
            **⚠️ Important:**
            - Results are AI-generated
            - Always verify with medical experts
            """)
        
        # Performance Metrics
        if is_active:
            with st.expander("📈 Model Information"):
                st.markdown("""
                **Architecture:** YOLOv8 Segmentation
                
                **Capabilities:**
                - Real-time detection
                - Pixel-level segmentation
                - Multi-class classification
                - Confidence scoring
                
                **Training:** Specialized medical imaging dataset
                """)
        
        # Timestamp
        st.markdown(f"<div style='text-align: center; color: #94a3b8; font-size: 0.75rem; margin-top: 2rem;'>Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)

    # -------------------------------------------------
    # MAIN INTERFACE
    # -------------------------------------------------
    st.markdown('<p class="section-header">📁 Upload Medical Scans</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Select one or multiple MRI scan images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        help="Supported formats: JPG, PNG, JPEG"
    )

    if not uploaded_files:
        st.markdown("""
        <div class="upload-section">
            <h3 style='color: #667eea; margin-bottom: 1rem;'>📤 Ready to Analyze</h3>
            <p style='color: #6b7280; font-size: 1.1rem;'>Upload MRI scans to begin AI-powered tumor detection and segmentation</p>
            <p style='color: #9ca3af; margin-top: 1rem;'>Drag and drop files or click to browse</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show features when no files uploaded
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card info">
                <h4 style='color: #3b82f6;'>🎯 High Accuracy</h4>
                <p style='color: #6b7280;'>Advanced deep learning model trained on thousands of medical images</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card success">
                <h4 style='color: #10b981;'>⚡ Real-time Analysis</h4>
                <p style='color: #6b7280;'>Get instant results with detailed segmentation and staging</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card warning">
                <h4 style='color: #f59e0b;'>📊 Comprehensive Reports</h4>
                <p style='color: #6b7280;'>Detailed metrics including tumor size, location, and confidence scores</p>
            </div>
            """, unsafe_allow_html=True)

    if uploaded_files and model:
        st.markdown("---")
        st.markdown(f'<p class="section-header">🔬 Analysis Results ({len(uploaded_files)} scan{"s" if len(uploaded_files) > 1 else ""})</p>', unsafe_allow_html=True)
        
        # Optional Patient Information
        with st.expander("📋 Patient Information (Optional)", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                patient_id = st.text_input("Patient ID", placeholder="e.g., P-2026-001")
            with col2:
                patient_age = st.number_input("Age", min_value=0, max_value=120, value=45)
            with col3:
                scan_date = st.date_input("Scan Date", value=datetime.now())
        
        # Batch Statistics Dashboard (if multiple files)
        if len(uploaded_files) > 1:
            st.markdown("### 📊 Batch Analysis Overview")
            
            # Pre-analyze all images for statistics
            all_results = []
            for uploaded_file in uploaded_files:
                img = Image.open(uploaded_file).convert("RGB")
                img_array = np.array(img)
                result = model.predict(img_array, conf=conf_threshold, verbose=False)[0]
                all_results.append({
                    'filename': uploaded_file.name,
                    'has_tumor': len(result.masks) > 0 if result.masks else False,
                    'num_tumors': len(result.masks) if result.masks else 0,
                    'result': result
                })
            
            # Calculate statistics
            total_scans = len(all_results)
            positive_scans = sum(1 for r in all_results if r['has_tumor'])
            negative_scans = total_scans - positive_scans
            total_tumors = sum(r['num_tumors'] for r in all_results)
            detection_rate = (positive_scans / total_scans * 100) if total_scans > 0 else 0
            
            # Display statistics cards
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{total_scans}</div>
                    <div class="stat-label">Total Scans</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_col2:
                st.markdown(f"""
                <div class="stat-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <div class="stat-number">{positive_scans}</div>
                    <div class="stat-label">Positive Cases</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_col3:
                st.markdown(f"""
                <div class="stat-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <div class="stat-number">{negative_scans}</div>
                    <div class="stat-label">Clean Scans</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_col4:
                st.markdown(f"""
                <div class="stat-box" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                    <div class="stat-number">{detection_rate:.0f}%</div>
                    <div class="stat-label">Detection Rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    tabs = st.tabs([f"📄 {f.name}" for f in uploaded_files])

    for i, uploaded_file in enumerate(uploaded_files):
        with tabs[i]:
            image = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(image)

        # Prediction
        with st.spinner("🔄 Analyzing scan with AI..."):
            results = model.predict(img_np, conf=conf_threshold)
            res = results[0]

        col_viz, col_data = st.columns([2.2, 1], gap="large")

        # -------------------------------------------------
        # VISUALIZATION COLUMN
        # -------------------------------------------------
        with col_viz:
            st.markdown("### 📊 Visual Analysis")

            viz_tabs = st.tabs([
                "🔍 Detection",
                "🎨 Segmentation Map",
                "🌡️ Grad-CAM"
            ])

            # 1️⃣ DETECTION VIEW
            with viz_tabs[0]:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(
                    res.plot(masks=False),
                    use_container_width=True,
                    caption="AI Object Detection - Tumor Localization"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("🔹 Bounding boxes indicate detected tumor regions with confidence scores")

            # 2️⃣ SEGMENTATION MAP (CLINICAL STYLE)
            with viz_tabs[1]:
                if res.masks:
                    img_overlay = img_np.copy()

                    for box, mask in zip(res.boxes, res.masks.xy):
                        poly = np.array(mask, dtype=np.int32)
                        area = int(cv2.contourArea(poly))

                        # Risk-based color coding (BGR)
                        if area < 5000:
                            border_color = (34, 197, 94)      # Green
                            fill_color = (34, 197, 94)
                            risk_level = "Low Risk"
                        elif area < 15000:
                            border_color = (245, 158, 11)     # Amber
                            fill_color = (245, 158, 11)
                            risk_level = "Moderate Risk"
                        else:
                            border_color = (239, 68, 68)      # Red
                            fill_color = (239, 68, 68)
                            risk_level = "High Risk"

                        # Semi-transparent fill overlay
                        overlay = img_overlay.copy()
                        cv2.fillPoly(overlay, [poly], fill_color)
                        img_overlay = cv2.addWeighted(
                            overlay, 0.18,
                            img_overlay, 0.82,
                            0
                        )

                        # Precise boundary marking
                        cv2.polylines(
                            img_overlay,
                            [poly],
                            True,
                            border_color,
                            border_thickness
                        )

                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(
                        img_overlay,
                        use_container_width=True,
                        caption="Precise Tumor Segmentation with Risk-Based Color Coding"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Color coding legend
                    legend_col1, legend_col2, legend_col3 = st.columns(3)
                    with legend_col1:
                        st.markdown("🟢 **Low Risk** - Area < 5,000 px²")
                    with legend_col2:
                        st.markdown("🟡 **Moderate** - Area 5,000-15,000 px²")
                    with legend_col3:
                        st.markdown("🔴 **High Risk** - Area > 15,000 px²")
                else:
                    st.info("✅ No tumor detected - Clean scan", icon="ℹ️")

                # 3️⃣ GRAD-CAM HEATMAP
                with viz_tabs[2]:
                    if res.masks is not None:
                        try:
                            mask = (
                                res.masks.data[0]
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )

                            mask_resized = cv2.resize(
                                mask,
                                (img_np.shape[1], img_np.shape[0])
                            )

                            heatmap = np.uint8(255 * mask_resized)
                            heatmap_color = cv2.applyColorMap(
                                heatmap,
                                cv2.COLORMAP_JET
                            )

                            grad_cam = cv2.addWeighted(
                                img_np,
                                0.6,
                                heatmap_color,
                                0.4,
                                0
                            )

                            st.markdown('<div class="image-container">', unsafe_allow_html=True)
                            st.image(
                                grad_cam,
                                use_container_width=True,
                                caption="Grad-CAM Visualization - AI Attention Map"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.caption("🔹 Warmer colors (red/yellow) indicate high model attention regions")
                        except Exception as e:
                            st.error(f"⚠️ Heatmap generation error: {e}", icon="🚨")
                    else:
                        st.info("✅ No tumor detected - Heatmap not applicable", icon="ℹ️")
                
                # Clinical Recommendations (under visualization)
                if res.masks:
                    st.markdown("---")
                    st.markdown("### 🏥 Clinical Recommendations")
                    # Get recommendations for the first/largest tumor
                    first_mask = res.masks.xy[0]
                    first_poly = np.array(first_mask, dtype=np.int32)
                    first_area = int(cv2.contourArea(first_poly))
                    first_label = model.names[int(res.boxes[0].cls[0])]
                    
                    if first_area < 5000:
                        risk_cat = "Low Risk"
                    elif first_area < 15000:
                        risk_cat = "Moderate Risk"
                    else:
                        risk_cat = "High Risk"
                    
                    recommendations = get_clinical_recommendation(risk_cat, first_area, first_label)
                    
                    rec_html = "<ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>"
                    for rec in recommendations:
                        rec_html += f"<li style='margin: 0.5rem 0; color: #475569;'>{rec}</li>"
                    rec_html += "</ul>"
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #fef3c7 0%, #fca5a5 100%); 
                                padding: 1.25rem; 
                                border-radius: 12px; 
                                border-left: 4px solid #f59e0b;'>
                        <h4 style='margin: 0 0 0.75rem 0; color: #92400e; font-size: 1rem;'>
                            ⚕️ Recommended Actions ({risk_cat})
                        </h4>
                        {rec_html}
                        <p style='margin: 0.75rem 0 0 0; font-size: 0.85rem; color: #92400e; font-style: italic;'>
                            ⚠️ These are automated suggestions. Always consult with qualified medical professionals.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            # -------------------------------------------------
            # CLINICAL REPORT COLUMN
            # -------------------------------------------------
            with col_data:
                st.markdown("### 📋 Clinical Report")

                if res.masks:
                    for j, (box, mask) in enumerate(
                        zip(res.boxes, res.masks.xy)
                    ):
                        label = model.names[int(box.cls[0])]
                        conf = float(box.conf[0])

                        poly = np.array(mask, dtype=np.int32)
                        area = int(cv2.contourArea(poly))

                        # Determine staging and card style
                        if area < 5000:
                            stage = "Stage I"
                            risk = "Low Risk"
                            card_class = "success"
                            badge_color = "#10b981"
                        elif area < 15000:
                            stage = "Stage II"
                            risk = "Moderate Risk"
                            card_class = "warning"
                            badge_color = "#f59e0b"
                        else:
                            stage = "Stage III"
                            risk = "High Risk"
                            card_class = "danger"
                            badge_color = "#ef4444"

                        # Professional card layout
                        st.markdown(f"""
                        <div class="metric-card {card_class}">
                            <div class="tumor-title">
                                🧬 Finding {j+1}: {label.upper()}
                            </div>
                            <span class="stage-badge" style="background: {badge_color}; color: white;">
                                {stage} - {risk}
                            </span>
                            <div style="margin-top: 1rem;">
                                <div class="metric-item">
                                    <span class="metric-label">Tumor Area</span>
                                    <span class="metric-value">{area:,} px²</span>
                                </div>
                                <div class="metric-item">
                                    <span class="metric-label">Confidence</span>
                                    <span class="metric-value">{conf:.1%}</span>
                                </div>
                                <div class="metric-item">
                                    <span class="metric-label">Classification</span>
                                    <span class="metric-value">{label.title()}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence bar
                        st.progress(conf, text=f"Detection Accuracy: {conf:.1%}")
                        
                        # Additional Metrics
                        st.markdown("##### 📐 Additional Measurements")
                        area_mm, volume_mm3 = calculate_tumor_volume(area)
                        st.markdown(f"""
                        <div style='background: #f8fafc; padding: 0.75rem; border-radius: 8px; margin-top: 0.5rem;'>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 0.3rem;'>
                                <span style='color: #64748b;'>Area (mm²):</span>
                                <span style='font-weight: 600; color: #1e293b;'>{area_mm:.1f}</span>
                            </div>
                            <div style='display: flex; justify-content: space-between;'>
                                <span style='color: #64748b;'>Est. Volume (mm³):</span>
                                <span style='font-weight: 600; color: #1e293b;'>{volume_mm3:.1f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                else:
                    st.markdown("""
                    <div class="clean-scan">
                        ✅ No Abnormalities Detected<br>
                        <small style="font-size: 0.9rem; opacity: 0.8;">Scan appears normal</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Download Section (outside the columns, full width)
            st.markdown("---")
            st.markdown("### 💾 Export Results")
            
            download_col1, download_col2, download_col3 = st.columns(3)
            
            with download_col1:
                # Download Detection Image
                detection_img = res.plot(masks=False)
                st.markdown(get_image_download_link(detection_img, f"detection_{uploaded_file.name}"), unsafe_allow_html=True)
                st.caption("Download detection view")
            
            with download_col2:
                # Download Segmentation if available
                if res.masks:
                    # Recreate segmentation image for download
                    seg_img = img_np.copy()
                    for box, mask in zip(res.boxes, res.masks.xy):
                        poly = np.array(mask, dtype=np.int32)
                        area = int(cv2.contourArea(poly))
                        if area < 5000:
                            color = (34, 197, 94)
                        elif area < 15000:
                            color = (245, 158, 11)
                        else:
                            color = (239, 68, 68)
                        overlay = seg_img.copy()
                        cv2.fillPoly(overlay, [poly], color)
                        seg_img = cv2.addWeighted(overlay, 0.18, seg_img, 0.82, 0)
                        cv2.polylines(seg_img, [poly], True, color, border_thickness)
                    
                    st.markdown(get_image_download_link(seg_img, f"segmentation_{uploaded_file.name}"), unsafe_allow_html=True)
                    st.caption("Download segmentation map")
                else:
                    st.info("No segmentation available")
            
            with download_col3:
                # Download original image
                st.markdown(get_image_download_link(img_np, f"original_{uploaded_file.name}"), unsafe_allow_html=True)
                st.caption("Download original scan")
    
    # Comparison Table for Multiple Scans
    if len(uploaded_files) > 1:
        st.markdown("---")
        st.markdown('<p class="section-header">📊 Comparative Analysis</p>', unsafe_allow_html=True)
        
        # Build comparison data
        comparison_data = []
        for idx, uploaded_file in enumerate(uploaded_files):
            img = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(img)
            result = model.predict(img_array, conf=conf_threshold, verbose=False)[0]
            
            if result.masks:
                num_tumors = len(result.masks)
                max_area = 0
                tumor_types = []
                avg_confidence = 0
                
                for box, mask in zip(result.boxes, result.masks.xy):
                    poly = np.array(mask, dtype=np.int32)
                    area = int(cv2.contourArea(poly))
                    max_area = max(max_area, area)
                    tumor_types.append(model.names[int(box.cls[0])])
                    avg_confidence += float(box.conf[0])
                
                avg_confidence = avg_confidence / num_tumors if num_tumors > 0 else 0
                
                if max_area < 5000:
                    risk = "🟢 Low"
                elif max_area < 15000:
                    risk = "🟡 Moderate"
                else:
                    risk = "🔴 High"
                
                status = "⚠️ Positive"
            else:
                num_tumors = 0
                max_area = 0
                tumor_types = ["-"]
                avg_confidence = 0
                risk = "✅ None"
                status = "✅ Negative"
            
            comparison_data.append({
                "Scan": uploaded_file.name,
                "Status": status,
                "Tumors Found": num_tumors,
                "Max Area (px²)": f"{max_area:,}" if max_area > 0 else "-",
                "Risk Level": risk,
                "Types": ", ".join(set(tumor_types)),
                "Avg. Confidence": f"{avg_confidence:.1%}" if avg_confidence > 0 else "-"
            })
        
        # Create DataFrame and display
        df = pd.DataFrame(comparison_data)
        
        # Custom styled table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Scan": st.column_config.TextColumn("Scan File", width="medium"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Tumors Found": st.column_config.NumberColumn("Tumors", width="small"),
                "Max Area (px²)": st.column_config.TextColumn("Max Area", width="small"),
                "Risk Level": st.column_config.TextColumn("Risk", width="small"),
                "Types": st.column_config.TextColumn("Type(s)", width="medium"),
                "Avg. Confidence": st.column_config.TextColumn("Confidence", width="small"),
            }
        )
        
        # Summary insights
        positive_count = sum(1 for d in comparison_data if "Positive" in d["Status"])
        high_risk_count = sum(1 for d in comparison_data if "🔴" in d["Risk Level"])
        
        insights_col1, insights_col2 = st.columns(2)
        with insights_col1:
            st.info(f"📌 **Key Finding:** {positive_count} out of {len(uploaded_files)} scans show abnormalities", icon="ℹ️")
        with insights_col2:
            if high_risk_count > 0:
                st.warning(f"⚠️ **Alert:** {high_risk_count} high-risk case(s) detected - Immediate review recommended", icon="🚨")
            else:
                st.success("✅ No high-risk cases detected in this batch", icon="✨")

    # -------------------------------------------------
    # FOOTER
    # -------------------------------------------------
    st.markdown("---")
    st.markdown(f"""
    <div class="footer">
        <div style="margin-bottom: 1rem;">
            <strong style="font-size: 1.1rem; color: #667eea;">NeuroScan AI Platform</strong>
        </div>
        <p style="margin: 0.5rem 0; color: #9ca3af;">
            Advanced Medical Imaging Analysis • Powered by YOLOv8 Deep Learning Architecture
        </p>
        <p style="margin: 0.5rem 0; color: #9ca3af; font-size: 0.85rem;">
            ⚠️ <strong>Disclaimer:</strong> This system is designed for research and educational purposes. 
            Always consult qualified medical professionals for diagnosis and treatment decisions.
        </p>
        <p style="margin-top: 1rem; font-size: 0.8rem; color: #cbd5e1;">
            © {datetime.now().year} NeuroScan AI • All Rights Reserved
        </p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# 9. PAGE ROUTING
# -------------------------------------------------
if page == "🔬 Analysis":
    render_analysis_page()
else:
    render_metrics_page()

