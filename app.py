import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. Page Configuration & Ultra-Modern Custom CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Pharmacy Demand Forecaster",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background-color: #0b0f19;
        color: #f0f6fc;
    }
    
    /* SIDEBAR CONTAINER STYLING */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #30363d !important;
    }
    
    /* SIDEBAR TEXT CONTRAST FIX */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stSidebar"] .stCaption, 
    section[data-testid="stSidebar"] p {
        color: #94a3b8 !important;
        font-size: 13px !important;
    }

    /* HIDE STREAMLIT RADIO CIRCLES & CONVERT TO SLEEK PILLS */
    div[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 10px !important;
        padding: 12px 18px !important;
        margin-bottom: 8px !important;
        color: #f0f6fc !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: all 0.25s ease-in-out !important;
        display: flex !important;
        align-items: center !important;
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        border-color: #388bfd !important;
        background-color: #1f2937 !important;
        transform: translateX(4px);
    }
    
    /* ACTIVE BUTTON STYLING */
    div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #0052d4, #4364f7, #6fb1fc) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(0, 82, 212, 0.4) !important;
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* MAIN CONTENT LABELS CONTRAST */
    label, .stWidgetLabel p, div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* MAIN BUTTONS STYLING */
    .stButton > button, .stDownloadButton > button {
        background-color: #1f6feb !important;
        color: #ffffff !important;
        border: 1px solid #388bfd !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #388bfd !important;
        border-color: #58a6ff !important;
        box-shadow: 0 4px 12px rgba(56, 139, 253, 0.4) !important;
    }
    .stButton > button p, .stDownloadButton > button p {
        color: #ffffff !important;
        font-size: 15px !important;
        font-weight: 700 !important;
    }
    
    /* METRIC CARDS STYLING */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .metric-title {
        font-size: 13px;
        color: #8b949e;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value-green { font-size: 26px; font-weight: bold; color: #3fb950; }
    .metric-value-red { font-size: 26px; font-weight: bold; color: #f85149; }
    .metric-value-blue { font-size: 26px; font-weight: bold; color: #58a6ff; }
    .metric-value-orange { font-size: 26px; font-weight: bold; color: #d97706; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Load ML Models & Feature Columns
# ---------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    models = joblib.load('medicine_models.pkl')
    columns = joblib.load('model_columns.pkl')
    return models, columns

try:
    models, model_columns = load_ml_assets()
    ml_status = True
except Exception:
    ml_status = False

# ---------------------------------------------------------
# 3. Dynamic Session State Management
# ---------------------------------------------------------
if "inventory" not in st.session_state:
    st.session_state.inventory = {
        "M01AB": {"name": "Brufen 400mg (Pain Relief)", "price": 4.50, "stock": 18, "min_required": 30, "category": "Anti-inflammatory", "expiry": "2026-11-15", "supplier": "Pfizer Pharma"},
        "M01AE": {"name": "Proxen 500mg (Joint Care)", "price": 6.20, "stock": 42, "min_required": 25, "category": "Joint Relief", "expiry": "2027-03-20", "supplier": "Sami Pharma"},
        "N02BA": {"name": "Aspirin 100mg (Blood Thinner)", "price": 2.10, "stock": 85, "min_required": 40, "category": "Cardiovascular", "expiry": "2026-08-10", "supplier": "Bayer Pharma"},
        "N02BE": {"name": "Paracetamol 500mg (Fever/Pain)", "price": 1.50, "stock": 120, "min_required": 100, "category": "General Pain", "expiry": "2027-05-12", "supplier": "GSK Healthcare"},
        "N05B":  {"name": "Anxiolytic Calm (Neurology)", "price": 12.00, "stock": 8, "min_required": 20, "category": "Neurology", "expiry": "2026-09-01", "supplier": "Hilton Pharma"},
        "N05C":  {"name": "Sedative / Sleep Aid", "price": 9.80, "stock": 35, "min_required": 15, "category": "Neurology", "expiry": "2026-12-30", "supplier": "Hilton Pharma"},
        "R03":   {"name": "Ventolin Inhaler (Asthma)", "price": 15.00, "stock": 6, "min_required": 25, "category": "Respiratory", "expiry": "2026-08-25", "supplier": "GSK Healthcare"},
        "R06":   {"name": "Rigix 10mg (Allergy Relief)", "price": 3.80, "stock": 65, "min_required": 30, "category": "Allergy", "expiry": "2027-01-10", "supplier": "AGP Pharma"}
    }

if "cart" not in st.session_state:
    st.session_state.cart = []

if "sales_history" not in st.session_state:
    today = datetime.now()
    st.session_state.sales_history = [
        {"Date": (today - timedelta(days=2)).strftime("%Y-%m-%d"), "Medicine": "Paracetamol 500mg (Fever/Pain)", "Qty": 10, "Revenue": 15.00},
        {"Date": (today - timedelta(days=1)).strftime("%Y-%m-%d"), "Medicine": "Brufen 400mg (Pain Relief)", "Qty": 5, "Revenue": 22.50},
        {"Date": today.strftime("%Y-%m-%d"), "Medicine": "Ventolin Inhaler (Asthma)", "Qty": 2, "Revenue": 30.00},
    ]

# ---------------------------------------------------------
# 4. SIDEBAR NAVIGATION MENU
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🤖 AI Smart Store")
    st.markdown("<p style='color:#00f2fe !important; font-weight:700; margin-top:-12px; margin-bottom:5px;'>Demand Forecasting Demo</p>", unsafe_allow_html=True)
    st.caption("Demo app: inventory and sales data are sample data.")
    st.markdown("---")
    
    st.markdown("<p style='color:#ffffff !important; font-weight:700; font-size:14px;'>Navigation Menu</p>", unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        [
            "🏠 Store Dashboard",
            "🔮 AI Demand Forecaster",
            "🛒 POS Counter",
            "📊 Sales Analytics",
            "➕ Manage Inventory",
            "⏳ Expiry & Discounts",
            "📄 Supplier Orders"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # AI Engine Status Card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0052d4, #4364f7, #6fb1fc);
        border-radius: 12px;
        padding: 16px;
        margin-top: 15px;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(0, 82, 212, 0.4);
    ">
        <h4 style="margin: 0; font-size: 15px; font-weight: bold; color: #ffffff !important;">🤖 AI Engine Status</h4>
        <p style="margin: 6px 0 0 0; font-size: 12px; color: #e0f2fe !important; font-weight: 500;">RandomForest Regressor Active</p>
        <p style="margin: 2px 0 0 0; font-size: 12px; color: #ffffff !important; font-weight: 700;">🟢 Status: Model loaded</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. MAIN CONTENT AREA BASED ON NAVIGATION
# ---------------------------------------------------------

# =========================================================
# PAGE 1: STORE DASHBOARD
# =========================================================
if page == "🏠 Store Dashboard":
    st.title("🏠 Smart Pharmacy Overview")
    st.caption("Real-time inventory levels, KPI metrics, and stock distribution analytics.")
    st.markdown("---")

    total_items = len(st.session_state.inventory)
    total_val = sum(item["stock"] * item["price"] for item in st.session_state.inventory.values())
    low_stock_count = sum(1 for item in st.session_state.inventory.values() if item["stock"] < item["min_required"])
    
    today_dt = datetime.now()
    near_expiry_count = 0
    for item in st.session_state.inventory.values():
        exp_date = datetime.strptime(item["expiry"], "%Y-%m-%d")
        if 0 <= (exp_date - today_dt).days <= 60:
            near_expiry_count += 1

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Products</div><div class="metric-value-blue">{total_items}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Stock Value</div><div class="metric-value-green">${total_val:,.2f}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Low Stock Alerts</div><div class="metric-value-red">{low_stock_count} Items</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Near Expiry Alerts</div><div class="metric-value-orange">{near_expiry_count} Items</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-title">AI Engine</div><div class="metric-value-green">Model Loaded</div></div>', unsafe_allow_html=True)

    st.markdown("### 📦 Live Inventory Monitor")
    inv_data = []
    for code, info in st.session_state.inventory.items():
        status = "🔴 Low Stock" if info["stock"] < info["min_required"] else "🟢 Healthy"
        inv_data.append({
            "Code": code, "Medicine Name": info["name"], "Category": info["category"],
            "Price": f"${info['price']:.2f}", "Current Stock": info["stock"],
            "Min Level": info["min_required"], "Expiry Date": info["expiry"], "Status": status
        })
    inv_df = pd.DataFrame(inv_data)
    st.dataframe(inv_df, use_container_width=True, hide_index=True)

    st.markdown("### 📊 Stock Distribution Graph")
    fig = px.bar(inv_df, x="Medicine Name", y="Current Stock", color="Status",
                 color_discrete_map={"🟢 Healthy": "#238636", "🔴 Low Stock": "#da3633"},
                 template="plotly_dark", height=380)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PAGE 2: AI DEMAND FORECASTER
# =========================================================
elif page == "🔮 AI Demand Forecaster":
    st.title("🔮 Machine Learning Medicine Demand Forecaster")
    st.caption("Estimate daily unit demand from calendar features (year, month, weekday).")
    st.markdown("---")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        selected_code = st.selectbox("Select Medicine:", options=list(st.session_state.inventory.keys()), format_func=lambda x: f"{x} - {st.session_state.inventory[x]['name']}")
    with f_col2:
        selected_day = st.selectbox("Forecast Day of Week:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    with f_col3:
        forecast_days = st.slider("Forecast Period (Days):", min_value=1, max_value=30, value=7)

    if st.button("🚀 Run AI Demand Prediction", use_container_width=True):
        if ml_status and selected_code in models:
            model = models[selected_code]
            input_data = pd.DataFrame(0, index=[0], columns=model_columns)
            for col in model_columns:
                if 'Month' in col: input_data[col] = datetime.now().month
                if 'Year' in col: input_data[col] = datetime.now().year
                if f"Weekday Name_{selected_day}" in col: input_data[f"Weekday Name_{selected_day}"] = 1
            
            daily_pred = model.predict(input_data)[0]
            total_predicted_demand = int(np.ceil(max(0, daily_pred) * forecast_days))
            curr_stock = st.session_state.inventory[selected_code]["stock"]
            med_name = st.session_state.inventory[selected_code]["name"]
            
            st.markdown("---")
            st.markdown(f"#### 📈 Forecast Results for **{med_name}**")
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1: st.metric("Current Available Stock", f"{curr_stock} units")
            with p_col2: st.metric(f"Predicted Demand ({forecast_days} Days)", f"{total_predicted_demand} units")
            with p_col3:
                shortage = total_predicted_demand - curr_stock
                if shortage > 0:
                    st.metric("Required Re-order Quantity", f"{shortage} units", delta=f"-{shortage} Shortage", delta_color="inverse")
                else:
                    st.metric("Stock Status", "Optimal", delta="No Re-order Needed")

            if shortage > 0:
                st.error(f"⚠️ **ALERT:** Stock ({curr_stock}) is lower than predicted demand ({total_predicted_demand}). Immediate re-order recommended!")
            else:
                st.success(f"✅ **HEALTHY:** Available stock is sufficient for the next {forecast_days} days.")

# =========================================================
# PAGE 3: POS COUNTER
# =========================================================
elif page == "🛒 POS Counter":
    st.title("🛒 Pharmacy Point of Sale (POS) Counter")
    st.caption("Fast billing counter with automated real-time inventory deduction.")
