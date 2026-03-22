import pandas as pd
from db import supabase

# -------------------------
# CONFIG
# -------------------------
FILE_PATH = r"E:\Irrigation\ten_crops_et0.xlsx"   #  change this
TABLE_NAME = "irrigation_data"
CHUNK_SIZE = 500


# -------------------------
# STEP 1: LOAD FILE
# -------------------------
print("📥 Loading Excel file...")
df = pd.read_excel(FILE_PATH)

# -------------------------
# STEP 2: CLEAN DATA
# -------------------------
print("🧹 Cleaning column names...")
df.columns = df.columns.str.lower().str.strip()

# Ensure required column exists
if "date" not in df.columns:
    raise Exception("❌ 'date' column not found after cleaning!")

# Convert date properly
print("📅 Converting date column...")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 🔥 Convert to string (IMPORTANT)
df["date"] = df["date"].dt.strftime("%Y-%m-%d")

# Optional: drop bad rows
df = df.dropna(subset=["date"])
df = df.where(pd.notnull(df), None)
# -------------------------
# STEP 3: DEBUG CHECK
# -------------------------
print("🔍 Data Preview:")
print(df.head())
print(f"✅ Total rows: {len(df)}")

if df.empty:
    raise Exception("❌ DataFrame is empty! Check your Excel file.")

# -------------------------
# STEP 4: CONVERT TO DICT
# -------------------------
data = df.to_dict(orient="records")

# -------------------------
# STEP 5: UPLOAD IN CHUNKS
# -------------------------
print("🚀 Uploading to Supabase...")

for i in range(0, len(data), CHUNK_SIZE):
    chunk = data[i:i + CHUNK_SIZE]

    try:
        supabase.table(TABLE_NAME).insert(chunk).execute()
        print(f"✅ Uploaded {i + len(chunk)} / {len(data)}")

    except Exception as e:
        print(f"❌ Error at chunk {i}: {e}")
        break

print("🎉 Upload complete!")