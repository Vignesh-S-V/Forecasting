import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Forecasting App", layout="wide")


# -----------------------------
# SAFE HELPERS
# -----------------------------
def safe_background_gradient(styler, subset=None, cmap="Blues"):
    """
    Apply background gradient only if matplotlib is available.
    If not available, return the styler without gradient.
    """
    try:
        import matplotlib  # noqa: F401
        return styler.background_gradient(cmap=cmap, subset=subset)
    except Exception:
        return styler


def safe_show_dataframe(df: pd.DataFrame, currency_cols=None, percent_cols=None, gradient_cols=None, height=350):
    """
    Safely display dataframe in Streamlit without crashing if matplotlib is absent.
    """
    currency_cols = currency_cols or []
    percent_cols = percent_cols or []
    gradient_cols = gradient_cols or []

    if df is None or df.empty:
        st.info("No data available.")
        return

    display_df = df.copy()

    try:
        styler = display_df.style

        fmt_dict = {}
        for col in currency_cols:
            if col in display_df.columns:
                fmt_dict[col] = "₹{:,.0f}"
        for col in percent_cols:
            if col in display_df.columns:
                fmt_dict[col] = "{:.2f}%"

        if fmt_dict:
            styler = styler.format(fmt_dict)

        valid_gradient_cols = [c for c in gradient_cols if c in display_df.columns]
        if valid_gradient_cols:
            styler = safe_background_gradient(styler, subset=valid_gradient_cols, cmap="Blues")

        st.dataframe(
            styler,
            use_container_width=True,
            hide_index=True,
            height=height,
        )
    except Exception:
        # Ultimate fallback: plain dataframe
        fallback_df = display_df.copy()

        for col in currency_cols:
            if col in fallback_df.columns:
                fallback_df[col] = fallback_df[col].apply(
                    lambda x: f"₹{x:,.0f}" if pd.notnull(x) else ""
                )

        for col in percent_cols:
            if col in fallback_df.columns:
                fallback_df[col] = fallback_df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
                )

        st.dataframe(
            fallback_df,
            use_container_width=True,
            hide_index=True,
            height=height,
        )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to avoid filtering / merge issues.
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_product_column(df: pd.DataFrame) -> str:
    """
    Try to detect product column automatically.
    """
    candidates = ["Product", "Item", "SKU", "Style", "Material Code", "Product Name"]
    for c in candidates:
        if c in df.columns:
            return c

    # fallback: try case-insensitive
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["product", "item", "sku", "style", "material code", "product name"]:
        if key in lower_map:
            return lower_map[key]

    raise ValueError("No product column found. Expected one of: Product / Item / SKU / Style / Material Code / Product Name")


def find_date_column(df: pd.DataFrame) -> str:
    """
    Detect date column if available.
    """
    candidates = ["Date", "Month", "Period", "Invoice Date"]
    for c in candidates:
        if c in df.columns:
            return c

    lower_map = {c.lower(): c for c in df.columns}
    for key in ["date", "month", "period", "invoice date"]:
        if key in lower_map:
            return lower_map[key]

    return None


def find_demand_column(df: pd.DataFrame) -> str:
    """
    Detect quantity / demand / sales column.
    """
    candidates = ["Demand", "Qty", "Quantity", "Sales Qty", "Forecast Base", "Sale Qty"]
    for c in candidates:
        if c in df.columns:
            return c

    lower_map = {c.lower(): c for c in df.columns}
    for key in ["demand", "qty", "quantity", "sales qty", "forecast base", "sale qty"]:
        if key in lower_map:
            return lower_map[key]

    raise ValueError("No demand/quantity column found.")


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    if col and col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def compute_simple_forecast(product_df: pd.DataFrame, demand_col: str, forecast_months: int) -> pd.DataFrame:
    """
    Simple forecast logic:
    Forecast = average of last 3 non-null demand values repeated for selected horizon.
    Replace this with your exact logic if needed.
    """
    temp = product_df.copy()

    demand_series = pd.to_numeric(temp[demand_col], errors="coerce").dropna()

    if len(demand_series) == 0:
        base = 0
    else:
        base = demand_series.tail(3).mean()

    forecast_values = [round(base, 2)] * forecast_months

    return pd.DataFrame({
        "Forecast Month": [f"Month {i+1}" for i in range(forecast_months)],
        "Forecast Qty": forecast_values
    })


def build_metrics(filtered_df: pd.DataFrame, demand_col: str, product_col: str) -> pd.DataFrame:
    """
    Summary metrics table.
    """
    if filtered_df.empty:
        return pd.DataFrame()

    tmp = filtered_df.copy()
    tmp[demand_col] = pd.to_numeric(tmp[demand_col], errors="coerce").fillna(0)

    metrics = pd.DataFrame({
        "Metric": [
            "Products Count",
            "Rows Count",
            "Total Demand",
            "Average Demand",
            "Max Demand",
            "Min Demand",
        ],
        "Value": [
            tmp[product_col].nunique(),
            len(tmp),
            tmp[demand_col].sum(),
            tmp[demand_col].mean(),
            tmp[demand_col].max(),
            tmp[demand_col].min(),
        ]
    })

    return metrics


def filter_data(df: pd.DataFrame, product_col: str, selected_products: list) -> pd.DataFrame:
    """
    Proper filtering for All / multi product selection.
    """
    if not selected_products or "All Products" in selected_products:
        return df.copy()

    return df[df[product_col].astype(str).isin([str(x) for x in selected_products])].copy()


# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.title("Forecasting Dashboard")

    uploaded_file = st.file_uploader("Upload Excel / CSV file", type=["xlsx", "xls", "csv"])

    if uploaded_file is None:
        st.info("Please upload a file to continue.")
        return

    # -----------------------------
    # READ FILE
    # -----------------------------
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"File read error: {e}")
        return

    if raw_df.empty:
        st.warning("Uploaded file is empty.")
        return

    df = normalize_columns(raw_df)

    # -----------------------------
    # DETECT IMPORTANT COLUMNS
    # -----------------------------
    try:
        product_col = find_product_column(df)
        demand_col = find_demand_column(df)
        date_col = find_date_column(df)
    except Exception as e:
        st.error(str(e))
        st.write("Available columns:", list(df.columns))
        return

    if date_col:
        df = ensure_datetime(df, date_col)

    # -----------------------------
    # SIDEBAR FILTERS
    # -----------------------------
    st.sidebar.header("Filters")

    product_list = sorted(df[product_col].dropna().astype(str).unique().tolist())
    product_options = ["All Products"] + product_list

    selected_products = st.sidebar.multiselect(
        "Select Product(s)",
        options=product_options,
        default=["All Products"],
    )

    forecast_months = st.sidebar.selectbox(
        "Forecast Horizon",
        options=[1, 2, 3, 6, 12],
        index=2
    )

    # IMPORTANT:
    # If user selects any specific product, automatically remove "All Products" effect
    if selected_products and "All Products" in selected_products and len(selected_products) > 1:
        selected_products = [p for p in selected_products if p != "All Products"]

    # -----------------------------
    # FILTER DATA
    # -----------------------------
    filtered_df = filter_data(df, product_col, selected_products)

    if filtered_df.empty:
        st.warning("No data found for selected filter.")
        return

    # -----------------------------
    # RAW DATA PREVIEW
    # -----------------------------
    st.subheader("Filtered Data")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    # -----------------------------
    # METRICS
    # -----------------------------
    st.subheader("Summary Metrics")
    metrics_df = build_metrics(filtered_df, demand_col, product_col)

    safe_show_dataframe(
        metrics_df,
        currency_cols=[],
        percent_cols=[],
        gradient_cols=["Value"],
        height=250
    )

    # -----------------------------
    # FORECAST SECTION
    # -----------------------------
    st.subheader("Forecast Results")

    products_for_forecast = (
        sorted(filtered_df[product_col].dropna().astype(str).unique().tolist())
    )

    all_forecasts = []

    for product_name in products_for_forecast:
        product_df = filtered_df[filtered_df[product_col].astype(str) == str(product_name)].copy()

        fcst_df = compute_simple_forecast(
            product_df=product_df,
            demand_col=demand_col,
            forecast_months=forecast_months,
        )

        fcst_df.insert(0, product_col, product_name)
        all_forecasts.append(fcst_df)

    if all_forecasts:
        final_forecast_df = pd.concat(all_forecasts, ignore_index=True)
    else:
        final_forecast_df = pd.DataFrame(columns=[product_col, "Forecast Month", "Forecast Qty"])

    st.dataframe(final_forecast_df, use_container_width=True, hide_index=True)

    # -----------------------------
    # CHART
    # -----------------------------
    st.subheader("Forecast Chart")

    chart_df = final_forecast_df.copy()

    if not chart_df.empty:
        fig = px.line(
            chart_df,
            x="Forecast Month",
            y="Forecast Qty",
            color=product_col,
            markers=True,
            title="Forecast by Product"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # DOWNLOAD
    # -----------------------------
    st.subheader("Download Forecast")
    csv_data = final_forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Forecast CSV",
        data=csv_data,
        file_name="forecast_output.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
