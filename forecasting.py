import io
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Forecasting App", layout="wide")


# =========================================================
# HELPERS
# =========================================================
def normalize_text(val):
    if pd.isna(val):
        return ""
    return str(val).strip()


def safe_read_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def detect_product_column(df):
    possible_product_cols = [
        "Product", "Product Name", "Item", "SKU", "Material", "Style", "Code"
    ]
    for c in possible_product_cols:
        if c in df.columns:
            return c
    return None


def detect_qty_column(df):
    possible_qty_cols = [
        "Qty", "Quantity", "Sales", "Demand", "Forecast Qty", "Value", "Order Qty"
    ]
    for c in possible_qty_cols:
        if c in df.columns:
            return c
    return None


def detect_date_column(df):
    possible_date_cols = [
        "Date", "Month", "Invoice Date", "Order Date", "Period", "Sales Date"
    ]
    for c in possible_date_cols:
        if c in df.columns:
            return c
    return None


def infer_frequency(series):
    s = pd.to_datetime(series, errors="coerce").dropna().sort_values()
    if len(s) < 3:
        return "MS"

    diffs = s.diff().dropna().dt.days
    if diffs.empty:
        return "MS"

    median_gap = diffs.median()

    if median_gap <= 2:
        return "D"
    if median_gap <= 10:
        return "W"
    if median_gap <= 35:
        return "MS"
    if median_gap <= 100:
        return "QS"
    return "YS"


def build_complete_series(product_df, date_col, qty_col, freq):
    temp = product_df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[qty_col] = pd.to_numeric(temp[qty_col], errors="coerce").fillna(0)

    temp = temp.dropna(subset=[date_col])
    temp = temp.groupby(date_col, as_index=False)[qty_col].sum()
    temp = temp.sort_values(date_col)

    if temp.empty:
        return temp

    min_date = temp[date_col].min()
    max_date = temp[date_col].max()

    full_index = pd.date_range(start=min_date, end=max_date, freq=freq)
    full_df = pd.DataFrame({date_col: full_index})

    merged = full_df.merge(temp, on=date_col, how="left")
    merged[qty_col] = merged[qty_col].fillna(0)

    return merged


def weighted_forecast(series, horizon):
    values = series.astype(float).tolist()
    if len(values) == 0:
        return [0.0] * horizon

    recent = values[-6:] if len(values) >= 6 else values[:]
    n = len(recent)
    weights = np.arange(1, n + 1, dtype=float)
    forecast_value = float(np.dot(recent, weights) / weights.sum()) if weights.sum() > 0 else 0.0
    forecast_value = max(0.0, forecast_value)

    return [forecast_value] * horizon


def simple_moving_average_forecast(series, horizon, window=3):
    values = series.astype(float).tolist()
    if len(values) == 0:
        return [0.0] * horizon

    w = min(window, len(values))
    avg = float(np.mean(values[-w:])) if w > 0 else 0.0
    avg = max(0.0, avg)
    return [avg] * horizon


def linear_trend_forecast(series, horizon):
    y = series.astype(float).values
    if len(y) == 0:
        return [0.0] * horizon
    if len(y) == 1:
        return [max(0.0, float(y[0]))] * horizon

    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)

    preds = []
    for i in range(1, horizon + 1):
        val = intercept + slope * (len(y) + i - 1)
        preds.append(max(0.0, float(val)))
    return preds


def choose_best_method(train_series, test_series):
    methods = {
        "Weighted Average": weighted_forecast,
        "Moving Average": simple_moving_average_forecast,
        "Linear Trend": linear_trend_forecast,
    }

    best_name = None
    best_preds = None
    best_mape = np.inf

    horizon = len(test_series)

    for name, func in methods.items():
        preds = func(train_series, horizon)
        preds = np.array(preds, dtype=float)
        actual = np.array(test_series, dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            ape = np.where(actual == 0, np.nan, np.abs((actual - preds) / actual) * 100)

        mape = np.nanmean(ape)
        if np.isnan(mape):
            mape = 999999

        if mape < best_mape:
            best_mape = mape
            best_name = name
            best_preds = preds

    return best_name, best_preds


def calculate_metrics(actual, forecast):
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)

    mae = np.mean(np.abs(actual - forecast)) if len(actual) > 0 else 0.0
    rmse = np.sqrt(np.mean((actual - forecast) ** 2)) if len(actual) > 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.where(actual == 0, np.nan, np.abs((actual - forecast) / actual) * 100)

    mape = np.nanmean(ape)
    if np.isnan(mape):
        mape = 0.0

    accuracy = max(0.0, 100.0 - mape)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "Accuracy": float(accuracy),
    }


def generate_forecast(product_data, date_col, qty_col, horizon, freq):
    series_df = build_complete_series(product_data, date_col, qty_col, freq)

    if series_df.empty:
        return pd.DataFrame(), {}

    y = series_df[qty_col].astype(float)

    # backtest split
    backtest_periods = min(3, max(1, len(y) // 4)) if len(y) >= 4 else 1

    if len(y) > backtest_periods:
        train_series = y.iloc[:-backtest_periods]
        test_series = y.iloc[-backtest_periods:]

        best_method, backtest_preds = choose_best_method(train_series, test_series)
        metrics = calculate_metrics(test_series.values, backtest_preds)
    else:
        best_method = "Weighted Average"
        metrics = {
            "MAE": 0.0,
            "RMSE": 0.0,
            "MAPE": 0.0,
            "Accuracy": 100.0,
        }

    # final fit on full data
    if best_method == "Weighted Average":
        future_preds = weighted_forecast(y, horizon)
    elif best_method == "Moving Average":
        future_preds = simple_moving_average_forecast(y, horizon, window=3)
    else:
        future_preds = linear_trend_forecast(y, horizon)

    last_date = pd.to_datetime(series_df[date_col].max())
    future_dates = pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq),
        periods=horizon,
        freq=freq
    )

    forecast_df = pd.DataFrame({
        date_col: future_dates,
        "Forecast_Qty": np.round(future_preds, 2)
    })

    metrics["Best_Method"] = best_method
    metrics["History_Periods"] = int(len(y))
    metrics["Last_Actual_Qty"] = float(y.iloc[-1]) if len(y) > 0 else 0.0
    metrics["Avg_History_Qty"] = float(np.mean(y)) if len(y) > 0 else 0.0

    return forecast_df, metrics


def to_excel_bytes(forecast_df, metrics_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, index=False, sheet_name="Forecast")
        metrics_df.to_excel(writer, index=False, sheet_name="Metrics")
    output.seek(0)
    return output.getvalue()


# =========================================================
# MAIN
# =========================================================
def main():
    st.title("Sales Forecasting App")
    st.write("Upload your CSV / Excel file and generate product-wise forecast.")

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is None:
        st.info("Please upload a file to continue.")
        return

    try:
        df = safe_read_file(uploaded_file)
    except Exception as e:
        st.error(f"File reading error: {e}")
        return

    if df.empty:
        st.error("Uploaded file is empty.")
        return

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    detected_date_col = detect_date_column(df)
    detected_product_col = detect_product_column(df)
    detected_qty_col = detect_qty_column(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        date_col = st.selectbox(
            "Select Date Column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(detected_date_col) if detected_date_col in df.columns else 0
        )

    with col2:
        product_col = st.selectbox(
            "Select Product Column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(detected_product_col) if detected_product_col in df.columns else 0
        )

    with col3:
        qty_col = st.selectbox(
            "Select Quantity / Demand Column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(detected_qty_col) if detected_qty_col in df.columns else 0
        )

    working_df = df.copy()

    working_df[product_col] = working_df[product_col].apply(normalize_text)
    working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce")
    working_df[qty_col] = pd.to_numeric(working_df[qty_col], errors="coerce").fillna(0)

    working_df = working_df.dropna(subset=[date_col])
    working_df = working_df[working_df[product_col] != ""]

    if working_df.empty:
        st.error("No valid records found after cleaning.")
        return

    freq = infer_frequency(working_df[date_col])

    st.subheader("Forecast Settings")
    c1, c2 = st.columns(2)

    with c1:
        horizon = st.slider("Forecast Periods", min_value=1, max_value=12, value=3)

    with c2:
        all_products = sorted(working_df[product_col].dropna().unique().tolist())
        selected_product = st.selectbox(
            "Select Product",
            options=["All Products"] + all_products
        )

    if selected_product == "All Products":
        filtered_df = working_df.copy()
        products_to_run = sorted(filtered_df[product_col].dropna().unique().tolist())
    else:
        filtered_df = working_df[working_df[product_col] == selected_product].copy()
        products_to_run = [selected_product]

    if filtered_df.empty:
        st.warning("No data available for selected product.")
        return

    st.subheader("Filtered Data Preview")
    st.dataframe(filtered_df.head(20), use_container_width=True)

    run_forecast = st.button("Generate Forecast", type="primary")

    if not run_forecast:
        return

    forecast_list = []
    metrics_list = []

    progress = st.progress(0)
    total_products = max(1, len(products_to_run))

    for idx, current_product in enumerate(products_to_run, start=1):
        product_data = filtered_df[filtered_df[product_col] == current_product].copy()

        if product_data.empty:
            progress.progress(min(idx / total_products, 1.0))
            continue

        forecast_df_product, metrics_row = generate_forecast(
            product_data=product_data,
            date_col=date_col,
            qty_col=qty_col,
            horizon=horizon,
            freq=freq
        )

        if not forecast_df_product.empty:
            forecast_df_product = forecast_df_product.copy()
            forecast_df_product[product_col] = current_product
            forecast_list.append(forecast_df_product)

        if metrics_row:
            metrics_row = dict(metrics_row)
            metrics_row[product_col] = current_product
            metrics_list.append(metrics_row)

        progress.progress(min(idx / total_products, 1.0))

    final_forecast_df = pd.concat(forecast_list, ignore_index=True) if forecast_list else pd.DataFrame()
    metrics_df = pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()

    if final_forecast_df.empty:
        st.warning("No forecast could be generated.")
        return

    # reorder columns
    forecast_cols = [product_col, date_col, "Forecast_Qty"]
    forecast_cols = [c for c in forecast_cols if c in final_forecast_df.columns]
    final_forecast_df = final_forecast_df[forecast_cols]

    if not metrics_df.empty:
        metric_order = [
            product_col, "Best_Method", "History_Periods", "Last_Actual_Qty",
            "Avg_History_Qty", "MAE", "RMSE", "MAPE", "Accuracy"
        ]
        metric_order = [c for c in metric_order if c in metrics_df.columns]
        metrics_df = metrics_df[metric_order]

    st.success("Forecast generated successfully.")

    st.subheader("Forecast Output")
    display_forecast_df = final_forecast_df.copy()
    for col in display_forecast_df.columns:
        if pd.api.types.is_numeric_dtype(display_forecast_df[col]):
            display_forecast_df[col] = pd.to_numeric(display_forecast_df[col], errors="coerce").round(2)

    st.dataframe(display_forecast_df, use_container_width=True, hide_index=True)

    st.subheader("Forecast Metrics")
    if not metrics_df.empty:
        display_metrics_df = metrics_df.copy()
        for col in display_metrics_df.columns:
            if pd.api.types.is_numeric_dtype(display_metrics_df[col]):
                display_metrics_df[col] = pd.to_numeric(display_metrics_df[col], errors="coerce").round(2)

        st.dataframe(display_metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info("No metrics available.")

    csv_bytes = final_forecast_df.to_csv(index=False).encode("utf-8")
    excel_bytes = to_excel_bytes(final_forecast_df, metrics_df)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="Download Forecast CSV",
            data=csv_bytes,
            file_name="forecast_output.csv",
            mime="text/csv"
        )

    with d2:
        st.download_button(
            label="Download Forecast + Metrics Excel",
            data=excel_bytes,
            file_name="forecast_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


if __name__ == "__main__":
    main()
