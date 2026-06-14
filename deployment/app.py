import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Vendor Invoice Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .title-gradient {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #64748b; font-size: 1.1rem; margin-bottom: 1.5rem; }
    .manual-approval-banner {
        padding: 1.2rem; border-radius: 8px;
        background-color: #3b1d20; /* Dark red */
        border: 1px solid #7d262b;
        color: #fca5a5; /* Light red text */
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 1rem;
        text-align: left;
    }
    .approved-banner {
        padding: 1.2rem; border-radius: 8px;
        background-color: #1b3b22; /* Dark green */
        border: 1px solid #267d3d;
        color: #a7f3d0; /* Light green text */
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 1rem;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths (pointing to root data and deployment/models directories) ──────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "sample_inventory.db"))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "models"))

# ── DB helper ──────────────────────────────────────────────────────────────────
def get_conn():
    return sqlite3.connect(DB_PATH)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    flag_model    = joblib.load(os.path.join(MODELS_DIR, "invoice_flagging_model.joblib"))
    flag_scaler   = joblib.load(os.path.join(MODELS_DIR, "invoice_flagging_scaler.joblib"))
    freight_model = joblib.load(os.path.join(MODELS_DIR, "freight_model.joblib"))
    return flag_model, flag_scaler, freight_model

try:
    flag_model, flag_scaler, freight_model = load_models()
except Exception as e:
    st.error(f"⚠️ Could not load model files: {e}")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='text-align:center;color:#6366f1;font-weight:700;'>NAVIGATOR</h2>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Go to page:",
    ["🚨 Invoice Manual Approval Prediction", "🚚 Freight Cost Estimator", "🔎 Database Explorer"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align:center;font-size:0.85rem;color:#64748b;'>
    <strong>Vendor Invoice Intelligence</strong><br>Supervised Learning Pipeline<br>v1.0.0
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page 1 – Invoice Manual Approval Prediction
# ─────────────────────────────────────────────────────────────────────────────
if page == "🚨 Invoice Manual Approval Prediction":
    st.markdown("## 🚨 Invoice Manual Approval Prediction")
    st.markdown("**Objective**: Predict whether a vendor invoice should be flagged for manual approval based on abnormal cost, freight, or delivery patterns.")
    st.markdown("")

    # Wrap inputs in a bordered container
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            invoice_quantity = st.number_input("📦 Invoice Quantity", min_value=1, value=50)
        with col2:
            invoice_dollars = st.number_input("💰 Invoice Dollars", min_value=0.0, value=352.95, step=10.0)
        with col3:
            total_item_dollars = st.number_input("💵 Total Item Dollars", min_value=0.0, value=2476.00, step=10.0)

        col4, col5, _ = st.columns([1, 1, 1])
        with col4:
            freight = st.number_input("🚚 Freight Cost", min_value=0.0, value=1.73, step=0.1)
        with col5:
            total_item_quantity = st.number_input("🛒 Total Item Quantity", min_value=1, value=162)

        st.markdown("")
        evaluate_clicked = st.button("⚖️ Evaluate Invoice Risk")

    if evaluate_clicked:
        # Hidden defaults for remaining model features
        days_po_to_invoice  = 5
        total_brands        = 2
        avg_receiving_delay = 4.5

        input_data = pd.DataFrame([[
            invoice_quantity, invoice_dollars, freight,
            days_po_to_invoice, total_brands,
            total_item_quantity, total_item_dollars, avg_receiving_delay
        ]], columns=[
            'invoice_quantity', 'invoice_dollars', 'Freight',
            'days_po_to_invoice', 'total_brands',
            'total_item_quantity', 'total_item_dollars', 'avg_receiving_delay'
        ])
        
        input_scaled  = flag_scaler.transform(input_data)
        ml_prediction = flag_model.predict(input_scaled)[0]

        # Display result matching the red/green banner styling
        if ml_prediction == 1:
            st.markdown("<div class='manual-approval-banner'>Invoice requires MANUAL APPROVAL</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='approved-banner'>Invoice is APPROVED</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page 2 – Freight Cost Estimator
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🚚 Freight Cost Estimator":
    st.markdown("<div class='title-gradient'>Freight Cost Estimator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Estimate shipping freight charges based on invoice quantities and amounts</div>", unsafe_allow_html=True)

    st.info("This model uses **Linear Regression** trained on vendor invoices (R²: **97.0%**, MAE: **$24.46**).")

    col_fr1, col_fr2 = st.columns([2, 1])
    with col_fr1:
        st.markdown("#### 📦 Shipment Details")
        quantity = st.number_input("Total Quantity (Units)", min_value=1,   value=250)
        dollars  = st.number_input("Total Order Value ($)",  min_value=0.0, value=5200.0, step=50.0)

        st.markdown("---")
        if st.button("🚚 Estimate Freight Cost"):
            pred = max(0.0, freight_model.predict(
                pd.DataFrame([[quantity, dollars]], columns=['Quantity', 'Dollars'])
            )[0])
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(99,102,241,0.1),rgba(168,85,247,0.1));
                        border:1px solid rgba(99,102,241,0.4);border-radius:16px;
                        padding:2rem;text-align:center;margin-top:1.5rem;'>
                <div style='font-size:1.1rem;color:#94a3b8;font-weight:600;text-transform:uppercase;'>Estimated Freight Cost</div>
                <div style='font-size:3.2rem;font-weight:800;color:#ffffff;margin-top:0.5rem;'>${pred:,.2f}</div>
                <div style='font-size:0.85rem;color:#64748b;margin-top:0.8rem;'>
                    Linear Regression model &bull; MAE: <strong>$24.46</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_fr2:
        st.markdown("#### 💡 Tips")
        st.info("""
        * **Dollars** and **Quantity** are the primary freight cost drivers.
        * Most accurate for mid-to-large orders (freight > $25).
        """)

# ─────────────────────────────────────────────────────────────────────────────
# Page 3 – Database Explorer
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔎 Database Explorer":
    st.markdown("<div class='title-gradient'>Database Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Browse vendor invoice records from sample dataset</div>", unsafe_allow_html=True)

    conn = get_conn()
    vendors = ["All Vendors"] + pd.read_sql_query(
        "SELECT DISTINCT VendorName FROM vendor_invoice ORDER BY VendorName", conn
    )['VendorName'].tolist()

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        vendor_filter = st.selectbox("Filter by Vendor:", vendors)
    with col_f2:
        search_po = st.text_input("Search PO Number:")

    query  = "SELECT VendorNumber, VendorName, PONumber, InvoiceDate, Quantity, Dollars, Freight, PayDate FROM vendor_invoice WHERE 1=1"
    params = []
    if vendor_filter != "All Vendors":
        query  += " AND VendorName = ?"; params.append(vendor_filter)
    if search_po.strip():
        query  += " AND PONumber = ?";   params.append(search_po.strip())
    query += " ORDER BY InvoiceDate DESC LIMIT 200"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    st.markdown(f"#### 📄 Showing {len(df)} records")
    if not df.empty:
        st.dataframe(
            df.style.format({'Dollars': '${:,.2f}', 'Freight': '${:,.2f}', 'Quantity': '{:,}'}),
            use_container_width=True
        )
    else:
        st.warning("No records match your search.")
