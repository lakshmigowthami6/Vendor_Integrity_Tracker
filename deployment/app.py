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

# Load custom styling for a premium Look & Feel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .title-gradient {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.25rem;
        color: #ffffff;
    }
    
    .flagged-banner {
        padding: 1rem;
        border-radius: 12px;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #f87171;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .normal-banner {
        padding: 1rem;
        border-radius: 12px;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #4ade80;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get database connection
def get_db_connection():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.abspath(os.path.join(base_dir, "..", "data", "inventory.db"))
    return sqlite3.connect(db_path)

# Load model artifacts
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    flag_model = joblib.load(os.path.join(models_dir, "invoice_flagging_model.joblib"))
    flag_scaler = joblib.load(os.path.join(models_dir, "invoice_flagging_scaler.joblib"))
    freight_model = joblib.load(os.path.join(models_dir, "freight_model.joblib"))
    
    return flag_model, flag_scaler, freight_model

try:
    flag_model, flag_scaler, freight_model = load_models()
except Exception as e:
    st.error(f"Error loading model files. Have you run the training script? Detail: {e}")
    st.stop()

# Sidebar navigation
st.sidebar.markdown("<h2 style='text-align: center; color: #6366f1; font-weight: 700;'>NAVIGATOR</h2>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Go to page:",
    ["🛡️ Invoice Flagging Tool", "🚚 Freight Cost Estimator", "🔎 Database Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; font-size: 0.85rem; color: #64748b;'>
    <strong>Vendor Invoice Intelligence</strong><br>
    Supervised Learning Pipeline<br>
    v1.0.0
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Page 1: Dashboard Overview
# -------------------------------------------------------------------------
if page == "🛡️ Invoice Flagging Tool":
    st.markdown("<div class='title-gradient'>Invoice Flagging Tool</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Evaluate invoice risk using ML Classification models alongside exact business rules</div>", unsafe_allow_html=True)
    
    st.markdown("""
    > [!NOTE]  
    > **Transparency Notice**: This dashboard runs the **Random Forest** machine learning model AND matches it directly against the exact business rule. 
    > E.g., An invoice is flagged if its billing differs from internal item totals by **> $5** or if the warehouse average receiving delay exceeds **10 days**.
    """)
    
    # Input panel
    st.markdown("#### 📝 Invoice Specifications")
    invoice_quantity = st.number_input("Invoice Quantity", min_value=1, value=15)
    invoice_dollars = st.number_input("Invoice Dollars ($)", min_value=0.0, value=250.0, step=10.0)
    total_item_dollars = st.number_input("Actual Items Dollars ($)", min_value=0.0, value=248.5, step=10.0)
    freight = st.number_input("Freight Charge ($)", min_value=0.0, value=12.5, step=1.0)
    
    # Default values for date and delay metrics (removed from UI)
    days_po_to_invoice = 5
    total_brands = 2
    total_item_quantity = invoice_quantity
    avg_receiving_delay = 4.5
    
    st.markdown("---")
    
    # Trigger classification
    if st.button("🛡️ Run Risk Assessment"):
        # 1. Exact Rule Check
        dollar_diff = abs(invoice_dollars - total_item_dollars)
        rule_flagged = (dollar_diff > 5) or (avg_receiving_delay > 10)
        
        # 2. Machine Learning Check
        input_data = pd.DataFrame([[
            invoice_quantity, invoice_dollars, freight,
            days_po_to_invoice, total_brands,
            total_item_quantity, total_item_dollars, avg_receiving_delay
        ]], columns=[
            'invoice_quantity', 'invoice_dollars', 'Freight', 
            'days_po_to_invoice', 'total_brands', 
            'total_item_quantity', 'total_item_dollars', 'avg_receiving_delay'
        ])
        
        input_scaled = flag_scaler.transform(input_data)
        ml_prediction = flag_model.predict(input_scaled)[0]
        ml_proba = flag_model.predict_proba(input_scaled)[0][1]
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("#### 🤖 Machine Learning Model (Random Forest)")
            if ml_prediction == 1:
                st.markdown(f"""
                <div class='flagged-banner'>
                    ⚠️ FLAGGED ANOMALY (Risk Prob: {ml_proba*100:.1f}%)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='normal-banner'>
                    ✅ NORMAL INVOICE (Risk Prob: {ml_proba*100:.1f}%)
                </div>
                """, unsafe_allow_html=True)
                
        with col_res2:
            st.markdown("#### ⚖️ Exact Business Rule Analysis")
            if rule_flagged:
                reasons = []
                if dollar_diff > 5:
                    reasons.append(f"Price discrepancy of ${dollar_diff:.2f} (> $5.00 limit)")
                if avg_receiving_delay > 10:
                    reasons.append(f"Receiving delay of {avg_receiving_delay:.1f} days (> 10.0 days limit)")
                
                st.markdown(f"""
                <div class='flagged-banner'>
                    ⚠️ FLAGGED ANOMALY<br>
                    <span style='font-size:0.85rem; font-weight:normal;'>Reason: {", ".join(reasons)}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='normal-banner'>
                    ✅ NORMAL INVOICE<br>
                    <span style='font-size:0.85rem; font-weight:normal;'>All specifications fall within healthy thresholds.</span>
                </div>
                """, unsafe_allow_html=True)
                
        # Education notice if they disagree
        if ml_prediction != rule_flagged:
            st.warning("""
            ⚠️ **Model Discrepancy Found!** The machine learning model prediction differs from the exact business rule. 
            This happens because the Random Forest model approximates the decision boundary and can misclassify inputs near threshold limits. 
            **Recommendation**: Deploy the rule-based checker for production audits as it offers 100% precision.
            """)

# -------------------------------------------------------------------------
# Page 3: Freight Cost Estimator
# -------------------------------------------------------------------------
elif page == "🚚 Freight Cost Estimator":
    st.markdown("<div class='title-gradient'>Freight Cost Estimator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Estimate shipping freight charges based on invoice quantities and amounts</div>", unsafe_allow_html=True)
    
    st.markdown("""
    > [!TIP]  
    > This model uses **Linear Regression** trained on vendor invoices to estimate shipping costs. 
    > It is highly accurate ($R^2$: **97.0%**) for typical orders.
    """)
    
    col_fr1, col_fr2 = st.columns([2, 1])
    
    with col_fr1:
        st.markdown("#### 📦 Shipment details")
        quantity = st.number_input("Total Quantity (Units)", min_value=1, value=250)
        dollars = st.number_input("Total Order Value ($)", min_value=0.0, value=5200.0, step=50.0)
        
        st.markdown("---")
        if st.button("🚚 Estimate Freight Cost"):
            input_features = pd.DataFrame([[quantity, dollars]], columns=['Quantity', 'Dollars'])
            pred_freight = freight_model.predict(input_features)[0]
            
            # Bound prediction to >= 0
            pred_freight = max(0.0, pred_freight)
            
            # Print result in a prominent gradient card
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.1)); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 16px; padding: 2rem; text-align: center; margin-top: 1.5rem;'>
                <div style='font-size: 1.1rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;'>Estimated Freight Cost</div>
                <div style='font-size: 3.2rem; font-weight: 800; color: #ffffff; margin-top: 0.5rem;'>${pred_freight:,.2f}</div>
                <div style='font-size: 0.85rem; color: #64748b; margin-top: 0.8rem;'>
                    Linear Regression baseline model • Mean Absolute Error (MAE): <strong>$24.46</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_fr2:
        st.markdown("#### 💡 Estimate Tips")
        st.info("""
        * **Dollars and Quantity** are the primary drivers of freight charges.
        * **Larger Orders** are much cheaper per unit, but command larger total freight fees.
        * **Accuracy Warning**: The model has a Mean Absolute Error of **$24.46**. This means predictions on very small orders (freight < $25) are less reliable on a relative scale than massive orders.
        """)

# -------------------------------------------------------------------------
# Page 4: Database Explorer
# -------------------------------------------------------------------------
elif page == "🔎 Database Explorer":
    st.markdown("<div class='title-gradient'>Database Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Inspect and search raw vendor invoice records in the database</div>", unsafe_allow_html=True)
    
    conn = get_db_connection()
    
    # Load unique vendor names for filter
    vendors = pd.read_sql_query("select distinct VendorName from vendor_invoice order by VendorName", conn)['VendorName'].tolist()
    vendors = ["All Vendors"] + vendors
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        vendor_filter = st.selectbox("Filter by Vendor:", vendors)
    with col_f2:
        search_po = st.text_input("Search PO Number (exact match):")
        
    query = "select VendorNumber, VendorName, PONumber, InvoiceDate, Quantity, Dollars, Freight, PayDate from vendor_invoice where 1=1"
    params = []
    
    if vendor_filter != "All Vendors":
        query += " and VendorName = ?"
        params.append(vendor_filter)
        
    if search_po.strip():
        query += " and PONumber = ?"
        params.append(search_po.strip())
        
    query += " order by InvoiceDate desc limit 200"
    
    df_results = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    st.markdown(f"#### 📄 Latest {len(df_results)} Invoice Records")
    if not df_results.empty:
        # Standard pandas display formatted beautifully
        st.dataframe(
            df_results.style.format({
                'Dollars': '${:,.2f}',
                'Freight': '${:,.2f}',
                'Quantity': '{:,}'
            }),
            use_container_width=True
        )
    else:
        st.warning("No records found matching the search criteria.")
