"""
Run this script ONCE locally to generate a sample database for Hugging Face deployment.
This creates a small subset of the real inventory.db that can be uploaded.
"""
import sqlite3
import os

def create_sample_db():
    src_db = os.path.join("data", "inventory.db")
    out_db = os.path.join("data", "sample_inventory.db")

    os.makedirs(os.path.dirname(out_db), exist_ok=True)

    print(f"Connecting to source: {src_db}")
    src_conn = sqlite3.connect(src_db)
    dst_conn = sqlite3.connect(out_db)

    src_cursor = src_conn.cursor()
    dst_cursor = dst_conn.cursor()

    # ---- vendor_invoice: all 5543 rows (small enough) ----
    print("Copying vendor_invoice...")
    src_cursor.execute("SELECT * FROM vendor_invoice")
    rows = src_cursor.fetchall()
    src_cursor.execute("PRAGMA table_info(vendor_invoice)")
    cols = [c[1] for c in src_cursor.fetchall()]
    col_types = {c[1]: c[2] for c in src_conn.execute("PRAGMA table_info(vendor_invoice)").fetchall()}

    dst_cursor.execute(f"DROP TABLE IF EXISTS vendor_invoice")
    dst_cursor.execute(f"""CREATE TABLE vendor_invoice (
        VendorNumber BIGINT, VendorName TEXT, InvoiceDate TEXT,
        PONumber BIGINT, PODate TEXT, PayDate TEXT,
        Quantity BIGINT, Dollars FLOAT, Freight FLOAT, Approval TEXT
    )""")
    dst_cursor.executemany(f"INSERT INTO vendor_invoice VALUES ({','.join(['?']*len(cols))})", rows)
    print(f"  Inserted {len(rows)} rows")

    # ---- purchases: sample 20,000 rows ----
    print("Copying purchases (20,000 rows)...")
    src_cursor.execute("SELECT * FROM purchases ORDER BY RANDOM() LIMIT 20000")
    rows = src_cursor.fetchall()
    dst_cursor.execute("DROP TABLE IF EXISTS purchases")
    dst_cursor.execute("""CREATE TABLE purchases (
        InventoryId TEXT, Store BIGINT, Brand BIGINT, Description TEXT,
        Size TEXT, VendorNumber BIGINT, VendorName TEXT,
        PONumber BIGINT, PODate TEXT, ReceivingDate TEXT,
        InvoiceDate TEXT, PayDate TEXT,
        PurchasePrice FLOAT, Quantity BIGINT, Dollars FLOAT, Classification BIGINT
    )""")
    dst_cursor.executemany(f"INSERT INTO purchases VALUES ({','.join(['?']*16)})", rows)
    print(f"  Inserted {len(rows)} rows")

    dst_conn.commit()
    src_conn.close()
    dst_conn.close()
    print(f"\nSample database created at: {out_db}")
    print(f"Size: {os.path.getsize(out_db) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    create_sample_db()
