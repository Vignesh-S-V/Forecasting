import pandas as pd
import numpy as np
import streamlit as st

# ============================================================
# SAFE HELPERS
# ============================================================

def normalize_text(val):
    if pd.isna(val):
        return ""
    return str(val).strip()

# ============================================================
# PRODUCT COLUMN IDENTIFICATION
# ============================================================

possible_product_cols = ["Product", "Product Name", "Item", "SKU", "Material", "Style"]
product_col = None

for c in possible_product_cols:
    if c in df.columns:
        product_col = c
        break

if product_col is None:
    st.error("Product column not found in uploaded file.")
    st.stop()

df = df.copy()
df[product_col] = df[product_col].apply(normalize_text)

# date_col should already exist in your old code
if date_col not in df.columns:
    st.error(f"Date column '{date_col}' not found.")
    st.stop()

# ============================================================
# PRODUCT FILTER
# ============================================================

all_products = sorted([p for p in df[product_col].dropna().unique().tolist() if p != ""])
product_options = ["All Products"] + all_products

selected_product = st.selectbox(
    "Select Product",
    options=product_options,
    index=0
)

if selected_product == "All Products":
    filtered_df = df.copy()
else:
    filtered_df = df[df[product_col] == selected_product].copy()

if filtered_df.empty:
    st.warning("No data available for selected product.")
    st.stop()

# ============================================================
# FORECAST EXECUTION
# ============================================================

forecast_results = []
metrics_results = []

if selected_product == "All Products":
    products_to_run = sorted(filtered_df[product_col].dropna().unique().tolist())
else:
    products_to_run = [selected_product]

for current_product in products_to_run:
    product_data = filtered_df[filtered_df[product_col] == current_product].copy()

    if product_data.empty:
        continue

    product_data = product_data.sort_values(by=date_col).copy()

    # ========================================================
    # KEEP YOUR EXISTING FORECAST FUNCTION / MODEL LOGIC HERE
    # Replace only the variable handling, not your model logic
    # ========================================================
    #
    # Expected output:
    # forecast_df_product -> DataFrame
    # metrics_dict -> dict
    #
    forecast_df_product, metrics_dict = generate_forecast(product_data)

    if forecast_df_product is not None and not forecast_df_product.empty:
        forecast_df_product = forecast_df_product.copy()
        forecast_df_product[product_col] = current_product
        forecast_results.append(forecast_df_product)

    if metrics_dict is not None:
        metrics_row = dict(metrics_dict)
        metrics_row[product_col] = current_product
        metrics_results.append(metrics_row)

final_forecast_df = pd.concat(forecast_results, ignore_index=True) if forecast_results else pd.DataFrame()
metrics_df = pd.DataFrame(metrics_results) if metrics_results else pd.DataFrame()

# ============================================================
# DISPLAY FORECAST TABLE
# ============================================================

if not final_forecast_df.empty:
    st.subheader("Forecast Output")
    display_forecast_df = final_forecast_df.copy()

    numeric_cols = display_forecast_df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        display_forecast_df[col] = pd.to_numeric(display_forecast_df[col], errors="coerce").round(2)

    st.dataframe(
        display_forecast_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No forecast output generated.")

# ============================================================
# DISPLAY METRICS TABLE - NO STYLER / NO MATPLOTLIB REQUIRED
# ============================================================

if not metrics_df.empty:
    st.subheader("Forecast Metrics")
    display_metrics_df = metrics_df.copy()

    numeric_cols = display_metrics_df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        display_metrics_df[col] = pd.to_numeric(display_metrics_df[col], errors="coerce").round(2)

    st.dataframe(
        display_metrics_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No metrics available.")
