import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# FONTS + DARK THEME + CUSTOM CSS
# =========================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* ---------- Global ---------- */
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(127,90,240,0.10) 0%, transparent 45%),
            radial-gradient(circle at 90% 10%, rgba(44,182,125,0.08) 0%, transparent 40%),
            radial-gradient(circle at top left, #1a1c2c 0%, #0e0f1a 55%, #0a0b12 100%);
        color: #e6e6f0;
    }
    .main { padding: 1.2rem 2.6rem 3rem; }
    * { font-family: 'Inter', 'Segoe UI', sans-serif; }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #0e0f18; }
    ::-webkit-scrollbar-thumb { background: #3a3d55; border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: #7f5af0; }

    /* ---------- Hero Header ---------- */
    .hero {
        background: linear-gradient(135deg, rgba(127,90,240,0.14), rgba(44,182,125,0.08));
        border: 1px solid rgba(127,90,240,0.28);
        border-radius: 22px;
        padding: 2rem 2.4rem;
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 40px rgba(0,0,0,0.35);
    }
    .hero::before {
        content: "";
        position: absolute; inset: 0;
        background: radial-gradient(circle at 100% 0%, rgba(127,90,240,0.25), transparent 55%);
        pointer-events: none;
    }
    .hero-eyebrow {
        display: inline-block;
        font-family: 'Poppins', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: #b9a3ff;
        background: rgba(127,90,240,0.15);
        border: 1px solid rgba(127,90,240,0.4);
        padding: 0.28rem 0.85rem;
        border-radius: 999px;
        margin-bottom: 0.9rem;
    }

    /* ---------- Headings ---------- */
    h1 {
        font-family: 'Poppins', sans-serif !important;
        background: linear-gradient(90deg, #a48cff, #7f5af0 40%, #2cb67d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.8px;
        font-size: 2.5rem !important;
        margin-bottom: 0.3rem !important;
    }
    h2, h3 {
        font-family: 'Poppins', sans-serif !important;
        color: #f5f5fa !important;
        font-weight: 700 !important;
        letter-spacing: -0.2px;
    }
    h4 { color: #eceaff !important; font-weight: 600 !important; }
    .subtitle {
        color: #9a9ab0;
        font-size: 1.08rem;
        margin-top: 0;
        max-width: 640px;
        line-height: 1.5;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14151f 0%, #0e0f18 100%);
        border-right: 1px solid #262838;
    }
    section[data-testid="stSidebar"] * { color: #e6e6f0 !important; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-family: 'Poppins', sans-serif !important;
    }
    .sidebar-brand {
        display: flex; align-items: center; gap: 0.6rem;
        padding-bottom: 1rem; margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .sidebar-brand .logo-dot {
        width: 38px; height: 38px; border-radius: 12px;
        background: linear-gradient(135deg, #7f5af0, #2cb67d);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem; flex-shrink: 0;
        box-shadow: 0 4px 14px rgba(127,90,240,0.4);
    }
    .sidebar-brand .brand-text { line-height: 1.15; }
    .sidebar-brand .brand-text b { font-family: 'Poppins', sans-serif; font-size: 1rem; }
    .sidebar-brand .brand-text span { font-size: 0.75rem; color: #8a8aa3; }

    /* ---------- Cards / Containers ---------- */
    .glass-card {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(127,90,240,0.4);
    }
    .metric-pill {
        display: inline-block;
        background: rgba(127, 90, 240, 0.15);
        border: 1px solid rgba(127, 90, 240, 0.4);
        color: #b9a3ff;
        padding: 0.32rem 0.95rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.18rem 0.3rem 0.18rem 0;
    }

    /* ---------- KPI strip (top of page) ---------- */
    .kpi-row { display: flex; gap: 1rem; margin-bottom: 1.6rem; flex-wrap: wrap; }
    .kpi-card {
        flex: 1 1 200px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(127,90,240,0.45);
    }
    .kpi-label {
        font-size: 0.78rem; letter-spacing: 0.6px; text-transform: uppercase;
        color: #8a8aa3; font-weight: 600; margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-family: 'Poppins', sans-serif; font-size: 1.6rem; font-weight: 800;
        background: linear-gradient(90deg, #a48cff, #2cb67d);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    /* ---------- Inputs ---------- */
    div[data-baseweb="select"] > div, .stNumberInput input, .stSlider {
        background-color: #1b1d2b !important;
        border-radius: 10px !important;
        color: #e6e6f0 !important;
    }
    div[data-baseweb="select"] > div {
        border: 1px solid rgba(255,255,255,0.09) !important;
        transition: border-color 0.15s ease;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(127,90,240,0.5) !important;
    }
    label, .stMarkdown p { color: #cfcfe0 !important; }

    /* ---------- Form Container ---------- */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.8rem 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.25);
    }
    div[data-testid="stForm"] .stSubheader, div[data-testid="stForm"] h3 {
        border-bottom: 1px solid rgba(255,255,255,0.08);
        padding-bottom: 0.5rem;
    }

    /* ---------- Buttons ---------- */
    .stButton>button, .stFormSubmitButton>button {
        width: 100%;
        background: linear-gradient(90deg, #7f5af0, #5f3dc4);
        color: white;
        font-family: 'Poppins', sans-serif;
        font-size: 16px;
        font-weight: 700;
        padding: 0.75rem;
        border-radius: 12px;
        border: none;
        margin-top: 0.6rem;
        box-shadow: 0 4px 18px rgba(127, 90, 240, 0.35);
        transition: all 0.2s ease-in-out;
        letter-spacing: 0.2px;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(127, 90, 240, 0.55);
        background: linear-gradient(90deg, #8f6bff, #6c4fd6);
    }
    .stFormSubmitButton>button {
        background: linear-gradient(90deg, #7f5af0, #2cb67d) !important;
        font-size: 18px !important;
        box-shadow: 0 6px 24px rgba(44,182,125,0.3) !important;
    }

    /* ---------- Prediction Boxes ---------- */
    .prediction-box {
        padding: 2.2rem 1.8rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .prediction-box .badge {
        display: inline-block;
        font-family: 'Poppins', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        margin-bottom: 0.6rem;
    }
    .churn-yes {
        background: linear-gradient(145deg, rgba(244,67,54,0.16), rgba(244,67,54,0.04));
        border: 1px solid rgba(244,67,54,0.5);
        box-shadow: 0 0 40px rgba(244,67,54,0.18);
    }
    .churn-yes .badge { background: rgba(244,67,54,0.18); color: #ff8a80; border: 1px solid rgba(244,67,54,0.4); }
    .churn-no {
        background: linear-gradient(145deg, rgba(76,175,80,0.16), rgba(76,175,80,0.04));
        border: 1px solid rgba(76,175,80,0.5);
        box-shadow: 0 0 40px rgba(76,175,80,0.18);
    }
    .churn-no .badge { background: rgba(76,175,80,0.18); color: #a9e3ac; border: 1px solid rgba(76,175,80,0.4); }

    /* ---------- Risk factor chips ---------- */
    .risk-chip {
        display: flex; align-items: center; gap: 0.5rem;
        background: rgba(244,67,54,0.08);
        border-left: 3px solid #f44336;
        padding: 0.6rem 0.9rem;
        border-radius: 10px;
        margin-bottom: 0.45rem;
        color: #ffb3ab;
        font-size: 0.92rem;
        font-weight: 500;
        transition: transform 0.15s ease;
    }
    .risk-chip:hover { transform: translateX(3px); }
    .ok-chip {
        display: flex; align-items: center; gap: 0.5rem;
        background: rgba(76,175,80,0.08);
        border-left: 3px solid #4caf50;
        padding: 0.6rem 0.9rem;
        border-radius: 10px;
        color: #a9e3ac;
        font-weight: 500;
    }

    /* ---------- Divider ---------- */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* ---------- Expander ---------- */
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
    }

    /* ---------- Footer ---------- */
    .app-footer {
        margin-top: 2.5rem;
        padding: 1.2rem 0 0.5rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        color: #6c6c85;
        font-size: 0.85rem;
    }
    .app-footer b { color: #b9a3ff; }
    </style>
    """, unsafe_allow_html=True)


# =========================================================
# LOAD MODEL + PREPROCESSOR
# =========================================================
@st.cache_resource
def load_models():
    """Load the trained model and preprocessor"""
    try:
        model_files = list(Path('models').glob('best_model_*.pkl'))
        if not model_files:
            st.error("❌ No trained model found! Please train the model first.")
            return None, None

        model_path = model_files[0]
        preprocessor_path = Path('models/preprocessor.pkl')

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        return model, preprocessor
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None


def preprocess_input(input_df, preprocessor):
    """Preprocess input data using the same pipeline as training"""
    df = input_df.copy()

    # Step 1: Handle missing values (if any)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if df['TotalCharges'].isnull().sum() > 0:
        df.loc[df['TotalCharges'].isnull(), 'TotalCharges'] = df['TotalCharges'].median()

    # Step 2: Feature Engineering
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[-1, 12, 24, 48, 73],
        labels=[0, 1, 2, 3],
        include_lowest=True
    )
    df['tenure_group'] = df['tenure_group'].astype(float).astype(int)

    df['avg_monthly_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['avg_monthly_per_tenure'] = df['avg_monthly_per_tenure'].replace([np.inf, -np.inf], 0)

    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity',
                     'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df['num_services'] = 0
    for col in service_cols:
        if col in df.columns:
            df['num_services'] += (df[col] == 'Yes').astype(int)

    # Step 3: Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Step 4: Encode features
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        if col in preprocessor.label_encoders:
            le = preprocessor.label_encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
        else:
            df[col] = pd.Categorical(df[col]).codes

    # Step 5: Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_per_tenure']
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    df[numerical_cols] = preprocessor.scaler.transform(df[numerical_cols])

    return df


def gauge_chart(churn_probability):
    """Dark-themed gauge showing churn probability"""
    color = "#f44336" if churn_probability >= 50 else "#4caf50"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_probability,
        number={'suffix': "%", 'font': {'size': 40, 'color': '#f5f5fa'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#9a9ab0', 'tickfont': {'color': '#9a9ab0'}},
            'bar': {'color': color},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(76,175,80,0.25)'},
                {'range': [30, 60], 'color': 'rgba(255,193,7,0.25)'},
                {'range': [60, 100], 'color': 'rgba(244,67,54,0.25)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.8,
                'value': churn_probability
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e6e6f0'}
    )
    return fig


def confidence_bar(no_churn_probability, churn_probability):
    fig = go.Figure(go.Bar(
        x=[no_churn_probability, churn_probability],
        y=['Will Stay', 'Will Churn'],
        orientation='h',
        marker=dict(color=['#2cb67d', '#f44336']),
        text=[f'{no_churn_probability:.1f}%', f'{churn_probability:.1f}%'],
        textposition='auto',
        textfont=dict(color='white', size=14)
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=True, color='#e6e6f0'),
        showlegend=False,
        font={'color': '#e6e6f0'}
    )
    return fig


# =========================================================
# MAIN APP
# =========================================================
def main():
    st.markdown("""
    <div class="hero">
        <span class="hero-eyebrow">✨ AI-Powered Retention Intelligence</span>
        <h1>🔮 Customer Churn Prediction System</h1>
        <p class="subtitle">Spot the customers about to walk away before they do — instant risk scoring,
        explainable factors, and a clear action plan for your retention team.</p>
    </div>
    """, unsafe_allow_html=True)

    model, preprocessor = load_models()
    if model is None or preprocessor is None:
        st.stop()

    st.markdown("""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">🤖 Model</div>
            <div class="kpi-value" style="font-size:1.15rem;">Logistic Regression</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">📈 ROC-AUC</div>
            <div class="kpi-value">0.846</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">🎯 Accuracy</div>
            <div class="kpi-value">80.41%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">⚖️ F1-Score</div>
            <div class="kpi-value">0.593</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="logo-dot">🔮</div>
            <div class="brand-text"><b>ChurnGuard AI</b><br><span>Retention Intelligence</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## ⚙️ Model Information")
        st.markdown("""
        <div class="glass-card">
        <span class="metric-pill">Logistic Regression</span><br>
        <span class="metric-pill">ROC-AUC 0.8458</span>
        <span class="metric-pill">Accuracy 80.41%</span>
        <span class="metric-pill">F1-Score 0.5929</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Fill in customer details on the right  
        2. Click **Predict Churn**  
        3. Review probability + gauge  
        4. Check key risk factors
        """)

        st.markdown("---")
        st.markdown("### 💡 Quick Test Profiles")

        if st.button("🔴 Load High-Risk Profile"):
            st.session_state.test_profile = "high_risk"
        if st.button("🟢 Load Low-Risk Profile"):
            st.session_state.test_profile = "low_risk"

        st.markdown("""
        <div class="app-footer" style="margin-top:2rem; text-align:left; padding-top:1rem;">
            Built with <b>Streamlit</b> + <b>Plotly</b><br>© 2026 ChurnGuard AI
        </div>
        """, unsafe_allow_html=True)

    if 'test_profile' not in st.session_state:
        st.session_state.test_profile = None

    if st.session_state.test_profile == "high_risk":
        defaults = {
            'gender': 'Female', 'senior_citizen': 'No', 'partner': 'No', 'dependents': 'No',
            'tenure': 3, 'phone_service': 'Yes', 'multiple_lines': 'No',
            'internet_service': 'Fiber optic', 'online_security': 'No', 'online_backup': 'No',
            'device_protection': 'No', 'tech_support': 'No', 'streaming_tv': 'Yes',
            'streaming_movies': 'Yes', 'contract': 'Month-to-month', 'paperless_billing': 'Yes',
            'payment_method': 'Electronic check', 'monthly_charges': 85.0, 'total_charges': 255.0
        }
    elif st.session_state.test_profile == "low_risk":
        defaults = {
            'gender': 'Male', 'senior_citizen': 'No', 'partner': 'Yes', 'dependents': 'Yes',
            'tenure': 48, 'phone_service': 'Yes', 'multiple_lines': 'Yes',
            'internet_service': 'Fiber optic', 'online_security': 'Yes', 'online_backup': 'Yes',
            'device_protection': 'Yes', 'tech_support': 'Yes', 'streaming_tv': 'Yes',
            'streaming_movies': 'Yes', 'contract': 'Two year', 'paperless_billing': 'No',
            'payment_method': 'Credit card (automatic)', 'monthly_charges': 105.0, 'total_charges': 5040.0
        }
    else:
        defaults = {
            'gender': 'Male', 'senior_citizen': 'No', 'partner': 'No', 'dependents': 'No',
            'tenure': 12, 'phone_service': 'Yes', 'multiple_lines': 'No',
            'internet_service': 'DSL', 'online_security': 'No', 'online_backup': 'No',
            'device_protection': 'No', 'tech_support': 'No', 'streaming_tv': 'No',
            'streaming_movies': 'No', 'contract': 'Month-to-month', 'paperless_billing': 'Yes',
            'payment_method': 'Electronic check', 'monthly_charges': 70.0, 'total_charges': 840.0
        }

    st.header("📝 Customer Information")
    with st.form("customer_form"):
            st.subheader("👤 Demographics")
            demo_col1, demo_col2, demo_col3 = st.columns(3)

            with demo_col1:
                gender = st.selectbox("Gender", ["Male", "Female"],
                                       index=0 if defaults['gender'] == "Male" else 1)
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"],
                                               index=0 if defaults['senior_citizen'] == "No" else 1)

            with demo_col2:
                partner = st.selectbox("Has Partner", ["No", "Yes"],
                                        index=0 if defaults['partner'] == "No" else 1)
                dependents = st.selectbox("Has Dependents", ["No", "Yes"],
                                           index=0 if defaults['dependents'] == "No" else 1)

            with demo_col3:
                tenure = st.slider("Tenure (months)", 0, 72, defaults['tenure'])

            st.markdown("---")
            st.subheader("💳 Account Information")
            acc_col1, acc_col2, acc_col3 = st.columns(3)

            contract_options = ["Month-to-month", "One year", "Two year"]
            with acc_col1:
                contract = st.selectbox("Contract Type", contract_options,
                                         index=contract_options.index(defaults['contract']))
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"],
                                                  index=0 if defaults['paperless_billing'] == "No" else 1)

            payment_options = ["Electronic check", "Mailed check",
                                "Bank transfer (automatic)", "Credit card (automatic)"]
            with acc_col2:
                payment_method = st.selectbox("Payment Method", payment_options,
                                               index=payment_options.index(defaults['payment_method']))

            with acc_col3:
                monthly_charges = st.number_input("Monthly Charges ($)",
                                                    min_value=0.0,
                                                    max_value=200.0,
                                                    value=defaults['monthly_charges'],
                                                    step=5.0)
                total_charges = st.number_input("Total Charges ($)",
                                                  min_value=0.0,
                                                  max_value=10000.0,
                                                  value=defaults['total_charges'],
                                                  step=50.0)

            st.markdown("---")
            st.subheader("📞 Services")
            serv_col1, serv_col2 = st.columns(2)

            with serv_col1:
                phone_service = st.selectbox("Phone Service", ["No", "Yes"],
                                              index=0 if defaults['phone_service'] == "No" else 1)
                multiple_lines = st.selectbox("Multiple Lines",
                                               ["No", "Yes", "No phone service"],
                                               index=["No", "Yes", "No phone service"].index(defaults['multiple_lines']))

                internet_options = ["DSL", "Fiber optic", "No"]
                internet_service = st.selectbox("Internet Service", internet_options,
                                                 index=internet_options.index(defaults['internet_service']))

            with serv_col2:
                online_security = st.selectbox("Online Security",
                                                ["No", "Yes", "No internet service"],
                                                index=["No", "Yes", "No internet service"].index(defaults['online_security']))
                online_backup = st.selectbox("Online Backup",
                                              ["No", "Yes", "No internet service"],
                                              index=["No", "Yes", "No internet service"].index(defaults['online_backup']))
                device_protection = st.selectbox("Device Protection",
                                                  ["No", "Yes", "No internet service"],
                                                  index=["No", "Yes", "No internet service"].index(defaults['device_protection']))

            serv_col3, serv_col4 = st.columns(2)

            with serv_col3:
                tech_support = st.selectbox("Tech Support",
                                             ["No", "Yes", "No internet service"],
                                             index=["No", "Yes", "No internet service"].index(defaults['tech_support']))

            with serv_col4:
                streaming_tv = st.selectbox("Streaming TV",
                                             ["No", "Yes", "No internet service"],
                                             index=["No", "Yes", "No internet service"].index(defaults['streaming_tv']))
                streaming_movies = st.selectbox("Streaming Movies",
                                                 ["No", "Yes", "No internet service"],
                                                 index=["No", "Yes", "No internet service"].index(defaults['streaming_movies']))

            submitted = st.form_submit_button("🔮 Predict Churn", width='stretch')

    st.markdown("---")
    st.header("🎯 Prediction Results")

    if submitted:
        st.session_state.test_profile = None

        input_data = pd.DataFrame({
            'customerID': ['PRED-001'],
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })

        try:
            processed_data = preprocess_input(input_data, preprocessor)

            with st.expander("🔍 Debug Info (Click to expand)"):
                st.write("**Processed data shape:**", processed_data.shape)
                st.write("**Processed columns:**", processed_data.columns.tolist())
                st.write("**Sample values:**")
                st.dataframe(processed_data.head())

            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0]

            churn_probability = prediction_proba[1] * 100
            no_churn_probability = prediction_proba[0] * 100

            # ---- Full-width 3-column results dashboard ----
            res_col1, res_col2, res_col3 = st.columns([1.1, 1, 1])

            with res_col1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box churn-yes">
                        <span class="badge">⚠️ HIGH RISK</span>
                        <h3 style="margin: 0.5rem 0; color:#f5f5fa;">Customer Likely to Churn</h3>
                        <h1 style="color: #f44336; margin: 0; font-size:3.2rem; -webkit-text-fill-color:#f44336; background:none;">{churn_probability:.1f}%</h1>
                        <p style="margin: 0.5rem 0; color:#cfcfe0;">Churn Probability</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.warning("**Recommendation:** Implement retention strategies immediately!")
                    st.markdown("""
                    **Suggested Actions:**
                    - Offer special discount or promotion
                    - Reach out to customer support team
                    - Propose contract upgrade with benefits
                    - Survey customer satisfaction
                    """)
                else:
                    st.markdown(f"""
                    <div class="prediction-box churn-no">
                        <span class="badge">✅ LOW RISK</span>
                        <h3 style="margin: 0.5rem 0; color:#f5f5fa;">Customer Likely to Stay</h3>
                        <h1 style="color: #4caf50; margin: 0; font-size:3.2rem; -webkit-text-fill-color:#4caf50; background:none;">{no_churn_probability:.1f}%</h1>
                        <p style="margin: 0.5rem 0; color:#cfcfe0;">Retention Probability</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.success("**Status:** Customer retention looks good!")
                    st.markdown("""
                    **Continue to:**
                    - Maintain service quality
                    - Engage with loyalty programs
                    - Regular satisfaction checks
                    """)

            with res_col2:
                st.markdown("#### 🌡️ Churn Risk Gauge")
                st.plotly_chart(gauge_chart(churn_probability), use_container_width=True)

                st.markdown("#### 📊 Confidence Breakdown")
                st.plotly_chart(confidence_bar(no_churn_probability, churn_probability), use_container_width=True)

            with res_col3:
                st.markdown("#### 🎯 Key Risk Factors")

                risk_factors = []
                if contract == "Month-to-month":
                    risk_factors.append("Month-to-month contract")
                if tenure < 12:
                    risk_factors.append("Short tenure (< 1 year)")
                if monthly_charges > 80:
                    risk_factors.append("High monthly charges")
                if online_security == "No":
                    risk_factors.append("No online security")
                if tech_support == "No":
                    risk_factors.append("No tech support")
                if payment_method == "Electronic check":
                    risk_factors.append("Electronic check payment")
                if partner == "No":
                    risk_factors.append("No partner")

                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f'<div class="risk-chip"><span>⚠️</span><span>{factor}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ok-chip"><span>✅</span><span>No major risk factors identified</span></div>',
                                unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        info_col1, info_col2 = st.columns([1, 1])
        with info_col1:
            st.info("👆 Fill in the customer details above and click **Predict Churn** to see results")
        with info_col2:
            st.markdown("""
            **💡 Quick Start** — Use the sidebar buttons to load test profiles:
            - 🔴 **High-Risk Profile**: New customer, month-to-month, no services
            - 🟢 **Low-Risk Profile**: Long tenure, 2-year contract, all services
            """)

    st.markdown("""
    <div class="app-footer">
        🔮 <b>ChurnGuard AI</b> — Predict churn before it happens · Powered by Streamlit, Plotly & Scikit-learn
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()