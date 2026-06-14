import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

def train_and_save_models():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.abspath(os.path.join(base_dir, "..", "data", "inventory.db"))
    models_dir = os.path.join(base_dir, "models")
    
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Connecting to database: {db_path}")
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return
        
    conn = sqlite3.connect(db_path)
    
    # -------------------------------------------------------------------------
    # 1. Train Invoice Flagging Model
    # -------------------------------------------------------------------------
    print("Training Invoice Flagging model...")
    df_flag = pd.read_sql_query("""
    with purchase_agg as(
        select p.PONumber,
        count(distinct Brand) as total_brands,
        sum(p.Quantity) as total_item_quantity,
        sum(p.Dollars) as total_item_dollars,
        avg(julianday(p.ReceivingDate)-julianday(p.PODate)) as avg_receiving_delay   
       from purchases p group by p.PONumber
    )
    select
        vi.PONumber,
        vi.Quantity as invoice_quantity, 
        vi.Dollars as invoice_dollars, vi.Freight, 
        (julianday(vi.InvoiceDate)-julianday(vi.PODate)) as days_po_to_invoice, 
        (julianday(vi.PayDate)-julianday(vi.InvoiceDate)) as days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
        from vendor_invoice vi
        left join purchase_agg pa on vi.PONumber = pa.PONumber
    """, conn)
    
    df_flag_clean = df_flag.dropna()
    
    def create_invoice_risk_label(row):
        if(abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5):
            return 1
        if row["avg_receiving_delay"] > 10:
            return 1
        return 0
        
    df_flag_clean['flag_invoice'] = df_flag_clean.apply(create_invoice_risk_label, axis=1)
    
    X_flag = df_flag_clean[[
        'invoice_quantity', 'invoice_dollars', 'Freight', 
        'days_po_to_invoice', 'total_brands', 
        'total_item_quantity', 'total_item_dollars', 'avg_receiving_delay'
    ]]
    y_flag = df_flag_clean['flag_invoice']
    
    # Scale features
    scaler = StandardScaler()
    X_flag_scaled = scaler.fit_transform(X_flag)
    
    # Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
    rf.fit(X_flag_scaled, y_flag)
    
    # Save model and scaler
    joblib.dump(rf, os.path.join(models_dir, "invoice_flagging_model.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "invoice_flagging_scaler.joblib"))
    print("Invoice Flagging model and scaler saved successfully.")
    
    # -------------------------------------------------------------------------
    # 2. Train Freight Prediction Model
    # -------------------------------------------------------------------------
    print("Training Freight Prediction model...")
    df_freight = pd.read_sql_query("select Quantity, Dollars, Freight from vendor_invoice", conn)
    df_freight_clean = df_freight.dropna()
    
    X_freight = df_freight_clean[['Quantity', 'Dollars']]
    y_freight = df_freight_clean['Freight']
    
    # Train Linear Regression model
    lr = LinearRegression()
    lr.fit(X_freight, y_freight)
    
    # Save model
    joblib.dump(lr, os.path.join(models_dir, "freight_model.joblib"))
    print("Freight Prediction model saved successfully.")
    
    conn.close()
    print("Model training completed.")

if __name__ == "__main__":
    train_and_save_models()
