import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Textile Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp { background: linear-gradient(135deg, #0f0c29, #1a1a4e, #0f0c29); color: #e8e8f0; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #13103a 0%, #1e1a5e 100%); border-right: 1px solid rgba(255,255,255,0.08); }
[data-testid="stSidebar"] * { color: #d4d4f0 !important; }

.metric-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; padding: 20px 24px; backdrop-filter: blur(10px); transition: all 0.3s ease; }
.metric-card:hover { border-color: rgba(120,100,255,0.4); background: rgba(255,255,255,0.08); }
.metric-label { font-size: 12px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: #8888cc; margin-bottom: 8px; }
.metric-value { font-size: 28px; font-weight: 700; color: #ffffff; font-family: 'JetBrains Mono', monospace; }
.metric-sub { font-size: 12px; color: #6666aa; margin-top: 4px; }

.section-header { font-size: 20px; font-weight: 700; color: #a09cf7; border-left: 4px solid #7c6fee; padding-left: 14px; margin: 28px 0 18px 0; letter-spacing: 0.5px; }

[data-testid="stFileUploader"] { background: rgba(120,100,255,0.08) !important; border: 2px dashed rgba(120,100,255,0.4) !important; border-radius: 14px !important; padding: 20px !important; }

.stButton > button { background: linear-gradient(135deg, #7c6fee, #9b8cf5) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; font-size: 15px !important; padding: 10px 28px !important; transition: all 0.3s ease !important; font-family: 'Space Grotesk', sans-serif !important; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(124,111,238,0.5) !important; }

.stSelectbox [data-baseweb="select"] { background: rgba(255,255,255,0.06) !important; border-color: rgba(255,255,255,0.15) !important; border-radius: 10px !important; }
.stAlert { border-radius: 12px !important; border: none !important; }

.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.04) !important; border-radius: 12px !important; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 9px !important; color: #9999cc !important; font-weight: 600 !important; }
.stTabs [aria-selected="true"] { background: rgba(124,111,238,0.35) !important; color: #ffffff !important; }

.outlier-badge { display: inline-block; background: rgba(255,80,80,0.2); border: 1px solid rgba(255,80,80,0.4); color: #ff8080; border-radius: 8px; padding: 3px 10px; font-size: 12px; font-weight: 600; }
.clean-badge { display: inline-block; background: rgba(80,200,120,0.2); border: 1px solid rgba(80,200,120,0.4); color: #60d080; border-radius: 8px; padding: 3px 10px; font-size: 12px; font-weight: 600; }

.data-warning-banner { background: linear-gradient(135deg, rgba(255,140,0,0.15), rgba(255,80,80,0.10)); border: 1px solid rgba(255,140,0,0.45); border-radius: 14px; padding: 16px 22px; margin: 12px 0 18px 0; }
.data-warning-banner .banner-title { font-size: 15px; font-weight: 700; color: #ffaa44; margin-bottom: 6px; }
.data-warning-banner .banner-body { font-size: 13px; color: #ccaa77; line-height: 1.6; }

.data-good-banner { background: linear-gradient(135deg, rgba(80,200,120,0.12), rgba(80,180,255,0.08)); border: 1px solid rgba(80,200,120,0.40); border-radius: 14px; padding: 16px 22px; margin: 12px 0 18px 0; }
.data-good-banner .banner-title { font-size: 15px; font-weight: 700; color: #50d090; margin-bottom: 6px; }
.data-good-banner .banner-body { font-size: 13px; color: #88ccaa; line-height: 1.6; }

.acc-badge-low { display:inline-block; background:rgba(255,80,80,0.18); border:1px solid rgba(255,80,80,0.45); color:#ff8080; border-radius:8px; padding:4px 12px; font-size:13px; font-weight:700; }
.acc-badge-mid { display:inline-block; background:rgba(255,165,0,0.18); border:1px solid rgba(255,165,0,0.45); color:#ffaa44; border-radius:8px; padding:4px 12px; font-size:13px; font-weight:700; }
.acc-badge-high { display:inline-block; background:rgba(80,200,120,0.18); border:1px solid rgba(80,200,120,0.45); color:#50d090; border-radius:8px; padding:4px 12px; font-size:13px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Space Grotesk", color="#d4d4f0"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showgrid=True),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", showgrid=True),
    margin=dict(l=60, r=40, t=60, b=50),
)
ACCENT = ["#7c6fee", "#f5a623", "#50d090", "#e05cf5", "#f55050"]

# ─────────────────────────────────────────────────────────────────────────────
# ★ DATA-LENGTH INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

MIN_MONTHS_FULL_ACCURACY = 18
MIN_MONTHS_BASIC         = 6


def data_length_status(n_months: int) -> dict:
    if n_months < MIN_MONTHS_BASIC:
        return dict(tier="insufficient", ok=False, icon="❌", label="Insufficient Data",
                    message=f"Only {n_months} months found. Minimum {MIN_MONTHS_BASIC} months required.",
                    use_full_models=False, season_len=max(2, n_months//2))
    elif n_months < MIN_MONTHS_FULL_ACCURACY:
        deficit = MIN_MONTHS_FULL_ACCURACY - n_months
        return dict(tier="limited", ok=True, icon="⚠️", label="Limited Accuracy",
                    message=(f"Only **{n_months} months** of data uploaded. "
                             f"Add **{deficit} more months** to unlock full accuracy mode. "
                             f"Seasonal patterns not fully captured — forecast may be less precise."),
                    use_full_models=False, season_len=min(n_months//2, 6))
    else:
        return dict(tier="full", ok=True, icon="✅", label="Full Accuracy Mode",
                    message=(f"**{n_months} months** of data — seasonal patterns fully captured. "
                             f"All models running at maximum accuracy."),
                    use_full_models=True, season_len=12)


def accuracy_tier_badge(mape: float) -> str:
    if mape < 5:
        return '<span class="acc-badge-high">🎯 Excellent (&lt;5% MAPE)</span>'
    elif mape < 10:
        return '<span class="acc-badge-high">✅ Good (&lt;10% MAPE)</span>'
    elif mape < 15:
        return '<span class="acc-badge-mid">⚠️ Fair (10–15% MAPE)</span>'
    else:
        return '<span class="acc-badge-low">❌ Poor (&gt;15% MAPE)</span>'


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_excel(file_bytes):
    return pd.read_excel(io.BytesIO(file_bytes))


def find_columns(df):
    col_map = {}; already_mapped = set()
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "date" and "Date" not in already_mapped:
            col_map[c] = "Date"; already_mapped.add("Date")
        elif cl in ("cy value","cyvalue","cy_value","sales","value","amount") and "Sales" not in already_mapped:
            col_map[c] = "Sales"; already_mapped.add("Sales")
        elif cl in ("subgrp","sub_grp","sub grp","productname","product_name","product name","product","category") and "SUBGRP" not in already_mapped:
            col_map[c] = "SUBGRP"; already_mapped.add("SUBGRP")
    return col_map


def safe_get_series(df, col_name):
    result = df[col_name]
    return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result


def deduplicate_columns(df):
    seen = {}; new_cols = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 0; new_cols.append(col)
        else:
            seen[col] += 1; new_cols.append(f"{col}__dup{seen[col]}")
    df.columns = new_cols
    return df


# ─────────────────────────────────────────────────────────────────────────────
# OUTLIER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_clean_outliers(monthly_series: pd.Series):
    y = monthly_series.values.astype(float); n = len(y)
    q1, q3 = np.percentile(y, 25), np.percentile(y, 75); iqr = q3 - q1
    iqr_lo, iqr_hi = q1 - 2.5*iqr, q3 + 2.5*iqr
    median = np.median(y); mad = np.median(np.abs(y - median))
    mad_lo, mad_hi = median - 3.5*1.4826*mad, median + 3.5*1.4826*mad
    mu, sigma = y.mean(), y.std()
    z_lo, z_hi = mu - 3.5*sigma, mu + 3.5*sigma
    flag_iqr = (y < iqr_lo) | (y > iqr_hi)
    flag_mad = (y < mad_lo) | (y > mad_hi)
    flag_z   = (y < z_lo)   | (y > z_hi)
    flags    = (flag_iqr.astype(int) + flag_mad.astype(int) + flag_z.astype(int)) >= 2
    y_clean = y.copy()
    for i in np.where(flags)[0]:
        left  = next((j for j in range(i-1,-1,-1) if not flags[j]), None)
        right = next((j for j in range(i+1,n)     if not flags[j]), None)
        if left is not None and right is not None:
            t = (i-left)/(right-left); y_clean[i] = y[left]*(1-t)+y[right]*t
        elif left  is not None: y_clean[i] = y[left]
        elif right is not None: y_clean[i] = y[right]
    summary = {"n_outliers": int(flags.sum()), "outlier_indices": list(np.where(flags)[0]),
                "iqr_bounds": (iqr_lo,iqr_hi), "mad_bounds": (mad_lo,mad_hi)}
    return pd.Series(y_clean, index=monthly_series.index), flags, summary


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

def holt_winters(series, season_len=12, forecast_periods=6):
    y = np.array(series, dtype=float); n = len(y)
    if n < 2*season_len: season_len = max(2, n//3)

    def ets_sse(params):
        alpha, beta, gamma = params
        if not (0 < alpha < 1 and 0 < beta < 1 and 0 < gamma < 1): return 1e15
        L = np.mean(y[:season_len])
        T = (np.mean(y[season_len:2*season_len]) - np.mean(y[:season_len]))/season_len if n >= 2*season_len else 0
        S = [y[i]-L for i in range(season_len)]; fitted = []; L_prev, T_prev = L, T
        for t in range(n):
            s_idx = t % season_len
            fitted.append((L_prev+T_prev+S[s_idx]) if t>0 else (L+T+S[s_idx]))
            L_new = alpha*(y[t]-S[s_idx]) + (1-alpha)*(L+T)
            T_new = beta*(L_new-L) + (1-beta)*T
            S[s_idx] = gamma*(y[t]-L_new) + (1-gamma)*S[s_idx]
            L_prev, T_prev = L, T; L, T = L_new, T_new
        return np.sum((y - np.array(fitted))**2)

    res = minimize(ets_sse, [0.3,0.1,0.3], method="Nelder-Mead", options={"maxiter":10000,"xatol":1e-6})
    alpha, beta, gamma = np.clip(res.x, 0.01, 0.99)
    L = np.mean(y[:season_len])
    T = (np.mean(y[season_len:2*season_len])-np.mean(y[:season_len]))/season_len if n>=2*season_len else 0
    S = [y[i]-L for i in range(season_len)]; fitted = []
    for t in range(n):
        s_idx = t % season_len; fitted.append(L+T+S[s_idx])
        L_new = alpha*(y[t]-S[s_idx])+(1-alpha)*(L+T)
        T_new = beta*(L_new-L)+(1-beta)*T
        S[s_idx] = gamma*(y[t]-L_new)+(1-gamma)*S[s_idx]; L, T = L_new, T_new
    forecast = [max(L+h*T+S[(n+h-1)%season_len], 0) for h in range(1, forecast_periods+1)]
    mape = np.mean(np.abs((y-np.array(fitted))/(y+1e-9)))*100
    return {"name":"Holt-Winters ETS","fitted":fitted,"forecast":forecast,"mape":mape,"params":{"alpha":alpha,"beta":beta,"gamma":gamma}}


def sarima_simple(series, forecast_periods=6, season=12):
    y = np.array(series, dtype=float); n = len(y)
    if n < season+3:
        fc = [y[-(season-i%season)] if season-i%season<=n else y[-1] for i in range(forecast_periods)]
        return {"name":"SARIMA","fitted":list(y),"forecast":fc,"mape":999,"params":{}}
    d1 = y[season:]-y[:-season]; d2 = np.diff(d1); p = min(2, len(d2)-1)
    if p < 1: p = 1
    X, Y_ar = [], []
    for i in range(p, len(d2)): X.append(d2[i-p:i][::-1]); Y_ar.append(d2[i])
    X, Y_ar = np.array(X), np.array(Y_ar)
    ar_coef = np.linalg.solve(X.T@X+1e-6*np.eye(p), X.T@Y_ar)
    d1_ext=list(d1); y_ext=list(y); d2_ext=list(d2); forecast=[]
    for _ in range(forecast_periods):
        next_d2=ar_coef@np.array(d2_ext[-p:])[::-1]; d2_ext.append(next_d2)
        next_d1=d1_ext[-1]+next_d2; d1_ext.append(next_d1)
        next_y=y_ext[-season]+next_d1; y_ext.append(max(next_y,0)); forecast.append(max(next_y,0))
    d2_fit=list(d2[:p]); fitted=list(y[:season+p])
    for i in range(p, len(d2)): d2_fit.append(ar_coef@np.array(d2_fit[-p:])[::-1])
    d1_fit=list(d1[:p])
    for v in d2_fit[p:]: d1_fit.append(d1_fit[-1]+v)
    y_fit=list(y[:season])
    for i,v in enumerate(d1_fit): y_fit.append(y_fit[i]+v)
    y_fit_arr=np.array(y_fit[:n])
    if len(y_fit_arr)<n: y_fit_arr=np.pad(y_fit_arr,(0,n-len(y_fit_arr)),"edge")
    mape=np.mean(np.abs((y-y_fit_arr)/(y+1e-9)))*100
    return {"name":"SARIMA","fitted":y_fit_arr.tolist(),"forecast":forecast,"mape":mape,"params":{"ar":ar_coef.tolist()}}


def seasonal_linear(series, forecast_periods=6, season=12):
    y=np.array(series,dtype=float); n=len(y); season=min(season,n//2)
    ma=np.convolve(y,np.ones(season)/season,mode="valid"); pad=(n-len(ma))//2
    trend=np.concatenate([np.full(pad,ma[0]),ma,np.full(n-pad-len(ma),ma[-1])])
    detrended=y-trend; months=np.array([i%season for i in range(n)])
    seas_idx=np.array([detrended[months==m].mean() for m in range(season)]); seas_idx-=seas_idx.mean()
    seasonal=np.array([seas_idx[i%season] for i in range(n)])
    slope,intercept,*_=stats.linregress(np.arange(n),y-seasonal)
    fitted=np.maximum(intercept+slope*np.arange(n)+seasonal,0)
    mape=np.mean(np.abs((y-fitted)/(y+1e-9)))*100
    forecast=[max(intercept+slope*(n-1+h)+seas_idx[(n-1+h)%season],0) for h in range(1,forecast_periods+1)]
    return {"name":"Seasonal Decomp + Trend","fitted":fitted.tolist(),"forecast":forecast,"mape":mape,"params":{"slope":slope,"intercept":intercept}}


def ml_poly_seasonal(series, forecast_periods=6, season=12):
    y=np.array(series,dtype=float); n=len(y); season=min(season,n//2)
    def build_feat(idx):
        t=np.array(idx,dtype=float)
        return np.column_stack([np.ones_like(t),t,t**2,
            np.sin(2*np.pi*t/season),np.cos(2*np.pi*t/season),
            np.sin(4*np.pi*t/season),np.cos(4*np.pi*t/season)])
    X=build_feat(range(n)); coef=np.linalg.solve(X.T@X+1e6*np.eye(X.shape[1]),X.T@y)
    fitted=np.maximum(X@coef,0); X_fut=build_feat(range(n,n+forecast_periods))
    forecast=np.maximum(X_fut@coef,0).tolist()
    mape=np.mean(np.abs((y-fitted)/(y+1e-9)))*100
    return {"name":"ML Poly+Seasonal (Ridge)","fitted":fitted.tolist(),"forecast":forecast,"mape":mape,"params":{"coef":coef.tolist()}}


def ensemble(models, forecast_periods=6):
    mapes=np.clip(np.array([m["mape"] for m in models]),0.01,None)
    weights=(1/mapes)/(1/mapes).sum()
    blend_fc=sum(w*np.array(m["forecast"]) for m,w in zip(models,weights))
    blend_ft=sum(w*np.array(m["fitted"])   for m,w in zip(models,weights))
    return {"name":"⭐ Weighted Ensemble","fitted":blend_ft.tolist(),"forecast":blend_fc.tolist(),
            "mape":float(np.average(mapes,weights=weights)),"params":{"weights":weights.tolist()}}


def confidence_intervals(forecast, residuals, alpha=0.10):
    std=np.std(residuals); z=stats.norm.ppf(1-alpha/2)
    return ([max(f-z*std*np.sqrt(h),0) for h,f in enumerate(forecast,1)],
            [f+z*std*np.sqrt(h) for h,f in enumerate(forecast,1)])


def compute_metrics(y_true, y_pred, name):
    y_true=np.array(y_true,dtype=float); y_pred=np.array(y_pred[:len(y_true)],dtype=float)
    mae=np.mean(np.abs(y_true-y_pred)); rmse=np.sqrt(np.mean((y_true-y_pred)**2))
    mape=np.mean(np.abs((y_true-y_pred)/(y_true+1e-9)))*100
    r2=1-np.sum((y_true-y_pred)**2)/(np.sum((y_true-np.mean(y_true))**2)+1e-9)
    return {"Model":name,"MAE (₹ Cr)":mae/1e7,"RMSE (₹ Cr)":rmse/1e7,"MAPE (%)":round(mape,2),"R²":round(r2,4)}


# ─────────────────────────────────────────────────────────────────────────────
# ★ SMART MODEL RUNNER — season_len adapts to data length
# ─────────────────────────────────────────────────────────────────────────────

def run_models(y, season_len, forecast_periods):
    models = []
    m1 = holt_winters(y, season_len=season_len, forecast_periods=forecast_periods)
    models.append(m1)
    if len(y) >= season_len + 3:
        m2 = sarima_simple(y, forecast_periods=forecast_periods, season=season_len)
        if m2["mape"] < 900: models.append(m2)
    m3 = seasonal_linear(y, forecast_periods=forecast_periods, season=season_len)
    models.append(m3)
    m4 = ml_poly_seasonal(y, forecast_periods=forecast_periods, season=season_len)
    models.append(m4)
    ens = ensemble(models, forecast_periods=forecast_periods)
    return models, ens


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_excel(df, ens, all_models, future_dates, metrics_df, lo, hi, subgrp):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    wb=Workbook(); thin=Border(left=Side(style="thin"),right=Side(style="thin"),top=Side(style="thin"),bottom=Side(style="thin"))
    def hdr(cell,bg="1A2C5B"):
        cell.font=Font(bold=True,color="FFFFFF",size=11); cell.fill=PatternFill("solid",start_color=bg)
        cell.alignment=Alignment(horizontal="center",vertical="center")
    ws1=wb.active; ws1.title="6-Month Forecast"
    hdrs1=["Month","Forecast (₹)","Forecast (₹ Cr)","Lower 90% CI (₹)","Upper 90% CI (₹)","MoM Change %"]
    for ci,h in enumerate(hdrs1,1): hdr(ws1.cell(1,ci,h)); ws1.column_dimensions[get_column_letter(ci)].width=22
    fc_arr=np.array(ens["forecast"]); last_actual=df["Sales"].values[-1]
    for i,(fd,fv,fl,fh_v) in enumerate(zip(future_dates,fc_arr,lo,hi)):
        prev=fc_arr[i-1] if i>0 else last_actual; mom=(fv-prev)/(prev+1e-9)*100
        row=[pd.Timestamp(fd).strftime("%b %Y"),fv,fv/1e7,fl,fh_v,mom/100]
        fmts=["@","#,##0","#,##0.00","#,##0","#,##0",'+0.0%;-0.0%']
        for ci,(val,fmt) in enumerate(zip(row,fmts),1):
            cell=ws1.cell(i+2,ci,val); cell.number_format=fmt
            cell.alignment=Alignment(horizontal="right" if ci>1 else "center"); cell.border=thin
    ws2=wb.create_sheet("Historical Data")
    hdrs2=["Month","Actual (₹)","Actual (₹ Cr)","Ensemble Fitted (₹)","Residual (₹)","MAPE Row %"]
    for ci,h in enumerate(hdrs2,1): hdr(ws2.cell(1,ci,h),"0D7377"); ws2.column_dimensions[get_column_letter(ci)].width=22
    fitted=np.array(ens["fitted"])
    for i,(date,actual) in enumerate(zip(df["Date"],df["Sales"])):
        residual=actual-fitted[i]; mrow=abs(residual)/(actual+1e-9)*100
        row=[date.strftime("%b %Y"),actual,actual/1e7,fitted[i],residual,mrow]; fmts=["@","#,##0","#,##0.00","#,##0","#,##0","0.00"]
        for ci,(val,fmt) in enumerate(zip(row,fmts),1):
            cell=ws2.cell(i+2,ci,val); cell.number_format=fmt
            cell.alignment=Alignment(horizontal="right" if ci>1 else "center"); cell.border=thin
    ws3=wb.create_sheet("Model Metrics")
    for ci,h in enumerate(metrics_df.columns,1): hdr(ws3.cell(1,ci,h),"4A148C"); ws3.column_dimensions[get_column_letter(ci)].width=24
    for ri,row_data in enumerate(metrics_df.itertuples(index=False),2):
        for ci,val in enumerate(row_data,1):
            cell=ws3.cell(ri,ci,val)
            if isinstance(val,float): cell.number_format="0.00"
            cell.alignment=Alignment(horizontal="center"); cell.border=thin
    buf=io.BytesIO(); wb.save(buf); return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_main_forecast(df, models, ens, future_dates, lo, hi):
    dates=df["Date"]; y_cr=df["Sales"]/1e7; fig=go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=y_cr,mode="lines+markers",name="Actual Sales",line=dict(color="#ffffff",width=2.5),marker=dict(size=6,color="#ffffff")))
    for m,c in zip(models,ACCENT):
        fig.add_trace(go.Scatter(x=future_dates,y=np.array(m["forecast"])/1e7,mode="lines+markers",name=f"{m['name']} (MAPE {m['mape']:.1f}%)",line=dict(color=c,width=1.5,dash="dot"),marker=dict(size=5),opacity=0.75))
    fig.add_trace(go.Scatter(x=list(future_dates)+list(future_dates)[::-1],y=list(np.array(hi)/1e7)+list(np.array(lo)/1e7)[::-1],fill="toself",fillcolor="rgba(230,80,150,0.12)",line=dict(color="rgba(255,255,255,0)"),hoverinfo="skip",showlegend=True,name="90% CI"))
    fig.add_trace(go.Scatter(x=future_dates,y=np.array(ens["forecast"])/1e7,mode="lines+markers",name=f"⭐ Ensemble (MAPE {ens['mape']:.1f}%)",line=dict(color="#e650a0",width=3),marker=dict(size=9,symbol="star")))
    fig.add_vline(x=str(df["Date"].max()),line_dash="dash",line_color="rgba(255,255,255,0.3)",line_width=1.5)
    fig.update_layout(title="📈 Sales Forecast — All Models",xaxis_title="Month",yaxis_title="Sales (₹ Crore)",legend=dict(orientation="h",yanchor="bottom",y=-0.25,x=0),height=480,**PLOTLY_THEME)
    return fig


def plot_outliers(df_monthly_raw, df_monthly_clean, flags, dates):
    fig=go.Figure(); y_raw=df_monthly_raw/1e7; y_clean=df_monthly_clean/1e7
    fig.add_trace(go.Scatter(x=dates,y=y_raw,mode="lines",name="Raw",line=dict(color="rgba(150,150,200,0.5)",width=1.5,dash="dot")))
    fig.add_trace(go.Scatter(x=dates,y=y_clean,mode="lines+markers",name="Cleaned",line=dict(color="#50d090",width=2.5),marker=dict(size=5,color="#50d090")))
    ol_dates=[d for d,f in zip(dates,flags) if f]; ol_vals=[v for v,f in zip(y_raw,flags) if f]
    if ol_dates: fig.add_trace(go.Scatter(x=ol_dates,y=ol_vals,mode="markers",name="Outlier Detected",marker=dict(size=12,color="#ff5050",symbol="x",line=dict(width=2,color="#ff8080"))))
    fig.update_layout(title="🔍 Outlier Detection & Data Cleaning",xaxis_title="Month",yaxis_title="Sales (₹ Crore)",height=380,**PLOTLY_THEME)
    return fig


def plot_seasonality_heatmap(df):
    df2=df.copy(); df2["Month"]=df2["Date"].dt.month; df2["Year"]=df2["Date"].dt.year; df2["Sales_Cr"]=df2["Sales"]/1e7
    pivot=df2.pivot_table(values="Sales_Cr",index="Year",columns="Month",aggfunc="sum")
    month_labels=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    col_labels=[month_labels[c-1] for c in pivot.columns]
    fig=go.Figure(go.Heatmap(z=pivot.values,x=col_labels,y=[str(y) for y in pivot.index],colorscale="YlOrRd",text=np.round(pivot.values,1),texttemplate="₹%{text}Cr",textfont=dict(size=9),hovertemplate="Year: %{y}<br>Month: %{x}<br>Sales: ₹%{z:.1f}Cr<extra></extra>",colorbar=dict(title="₹ Cr")))
    fig.update_layout(title="🗓️ Monthly Seasonality Heatmap (₹ Crore)",height=320,**PLOTLY_THEME)
    return fig


def plot_model_accuracy(all_models):
    names=[m["name"].replace("⭐ ","") for m in all_models]; mapes=[m["mape"] for m in all_models]; colors=ACCENT+["#e650a0"]
    fig=go.Figure(go.Bar(x=mapes,y=names,orientation="h",marker=dict(color=colors[:len(names)],opacity=0.85),text=[f"{v:.1f}%" for v in mapes],textposition="outside"))
    fig.add_vline(x=5,line_dash="dash",line_color="rgba(80,200,120,0.5)",annotation_text="Good (5%)")
    fig.add_vline(x=10,line_dash="dash",line_color="rgba(245,166,35,0.5)",annotation_text="Fair (10%)")
    fig.update_layout(title="🎯 Model Accuracy (MAPE — lower is better)",xaxis_title="MAPE %",height=340,**PLOTLY_THEME)
    return fig


def plot_detailed_forecast(df, ens, future_dates, lo, hi):
    tail_dates=df["Date"].values[-6:]; tail_actual=df["Sales"].values[-6:]/1e7
    fc_cr=np.array(ens["forecast"])/1e7; lo_cr=np.array(lo)/1e7; hi_cr=np.array(hi)/1e7
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=tail_dates,y=tail_actual,mode="lines+markers",name="Last 6M Actual",line=dict(color="#aaaaee",width=2),marker=dict(size=7,color="#aaaaee")))
    fig.add_trace(go.Scatter(x=list(future_dates)+list(future_dates)[::-1],y=list(hi_cr)+list(lo_cr)[::-1],fill="toself",fillcolor="rgba(230,80,150,0.15)",line=dict(color="rgba(0,0,0,0)"),name="90% CI",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_dates,y=fc_cr,mode="lines+markers+text",name="⭐ Ensemble Forecast",line=dict(color="#e650a0",width=3),marker=dict(size=10,symbol="star",color="#e650a0"),text=[f"₹{v:.1f}Cr" for v in fc_cr],textposition="top center",textfont=dict(size=11,color="#f080c0")))
    fig.update_layout(title="📅 Next 6-Month Detailed Forecast with 90% Confidence Interval",xaxis_title="Month",yaxis_title="Sales (₹ Crore)",height=420,**PLOTLY_THEME)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size:42px; font-weight:800; background: linear-gradient(135deg,#a09cf7,#f5a623,#50d090);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-1px;">
            📈 Textile Sales Forecasting
        </div>
        <div style="color:#7777aa; font-size:15px; margin-top:8px; letter-spacing:1px;">
            Upload → Filter by SUBGRP → Get 6-Month AI Forecast
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")
        uploaded = st.file_uploader("📂 Upload Excel File", type=["xlsx","xls"],
                                    help="File must have: Date, CY Value, SUBGRP columns")
        subgrp_choice=None; remove_outliers=True; forecast_periods=6

        if uploaded:
            st.success("✅ File uploaded!")
            try:
                raw_df=load_excel(uploaded.read())
                col_map=find_columns(raw_df)
                raw_df_renamed=deduplicate_columns(raw_df.rename(columns=col_map).copy())
                if "SUBGRP" in raw_df_renamed.columns:
                    subgrp_series=safe_get_series(raw_df_renamed,"SUBGRP")
                    subgrps=sorted(subgrp_series.dropna().unique().tolist())
                    subgrp_choice=st.selectbox("🏷️ Filter by SUBGRP",options=["All Products"]+subgrps,index=0)
                else:
                    st.warning("⚠️ No SUBGRP column found. Using all data.")
                st.markdown("---")
                remove_outliers=st.toggle("🔍 Remove Outliers",value=True)
                forecast_periods=st.slider("📅 Forecast Months",3,12,6)
                st.markdown("---")
                st.markdown("**📋 File Info**")
                st.markdown(f"- Rows: **{len(raw_df):,}**")
                st.markdown(f"- Columns: **{list(raw_df.columns)}**")
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                import traceback; st.code(traceback.format_exc()); return

        st.markdown("---")
        st.markdown("""
        <div style="font-size:11px; color:#555588; text-align:center;">
        Expected columns:<br><code>Date · CY Value · SUBGRP</code><br><br>
        <b>💡 Pro tip:</b> Upload <b>≥18 months</b><br>for full accuracy mode
        </div>""", unsafe_allow_html=True)

    # ── No file ───────────────────────────────────────────────────────────────
    if uploaded is None:
        col1,col2,col3=st.columns(3)
        for col,icon,title,desc in [
            (col1,"📂","Upload","Upload your Excel file with Date, CY Value, SUBGRP columns"),
            (col2,"🏷️","Filter","Select any SUBGRP product to forecast individually"),
            (col3,"🔮","Forecast","Get 6-month predictions with 4 AI models + ensemble"),
        ]:
            with col:
                st.markdown(f"""<div class="metric-card" style="text-align:center; padding:32px 20px;">
                    <div style="font-size:36px;">{icon}</div>
                    <div style="font-size:18px; font-weight:700; color:#a09cf7; margin:12px 0 8px;">{title}</div>
                    <div style="font-size:13px; color:#666699;">{desc}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="data-warning-banner">
            <div class="banner-title">📏 Data Length Guide — Accuracy Tiers</div>
            <div class="banner-body">
            &lt; 6 months &nbsp;→&nbsp; ❌ Cannot forecast<br>
            6 – 17 months &nbsp;→&nbsp; ⚠️ <b>Limited accuracy</b> — seasonal patterns incomplete<br>
            ≥ 18 months &nbsp;→&nbsp; ✅ <b>Full accuracy mode</b> — maximum forecast precision
            </div>
        </div>""", unsafe_allow_html=True)
        st.info("👈 Upload your Excel file from the sidebar to get started!")
        return

    # ── Process data ──────────────────────────────────────────────────────────
    try:
        raw_df=load_excel(uploaded.getvalue()); col_map=find_columns(raw_df)
        if "Date"  not in col_map.values(): st.error("❌ 'Date' column not found!"); return
        if "Sales" not in col_map.values(): st.error("❌ 'CY Value' / 'Sales' column not found!"); return
        df_work=deduplicate_columns(raw_df.rename(columns=col_map).copy())
        df_work["Date"] =pd.to_datetime(safe_get_series(df_work,"Date"),errors="coerce")
        df_work["Sales"]=pd.to_numeric(safe_get_series(df_work,"Sales"),errors="coerce").fillna(0)
        df_work=df_work.dropna(subset=["Date","Sales"])
        filter_label="All Products"
        if subgrp_choice and subgrp_choice!="All Products" and "SUBGRP" in df_work.columns:
            subgrp_col=safe_get_series(df_work,"SUBGRP")
            df_work=df_work[subgrp_col==subgrp_choice].copy(); filter_label=subgrp_choice
        if len(df_work)==0: st.error("❌ No data after filtering!"); return
        df_monthly=(df_work.groupby(df_work["Date"].dt.to_period("M"))["Sales"].sum().reset_index())
        df_monthly["Date"]=df_monthly["Date"].dt.to_timestamp()
        df_monthly=df_monthly.sort_values("Date").reset_index(drop=True)
        if len(df_monthly)<MIN_MONTHS_BASIC:
            st.error(f"❌ Only {len(df_monthly)} months. Minimum {MIN_MONTHS_BASIC} required."); return

        # ★ Data-length check
        dls=data_length_status(len(df_monthly))
        raw_sales=df_monthly["Sales"].copy()
        cleaned_sales,outlier_flags,outlier_summary=detect_and_clean_outliers(df_monthly["Sales"])
        df_forecast=df_monthly.copy()
        if remove_outliers: df_forecast["Sales"]=cleaned_sales
    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        import traceback; st.code(traceback.format_exc()); return

    # ── Data-length banner ────────────────────────────────────────────────────
    banner_cls = "data-good-banner" if dls["tier"]=="full" else "data-warning-banner"
    st.markdown(f"""
    <div class="{banner_cls}">
        <div class="banner-title">{dls['icon']} {dls['label']} — {len(df_monthly)} Months Loaded</div>
        <div class="banner-body">{dls['message']}</div>
    </div>""", unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-header'>📊 Data Overview — {filter_label}</div>", unsafe_allow_html=True)
    k1,k2,k3,k4,k5=st.columns(5)
    avg_cr=df_forecast["Sales"].mean()/1e7; peak_cr=df_forecast["Sales"].max()/1e7
    peak_m=df_forecast.loc[df_forecast["Sales"].idxmax(),"Date"].strftime("%b %Y")
    total_cr=df_forecast["Sales"].sum()/1e7
    df_forecast["Year"]=df_forecast["Date"].dt.year; df_forecast["Month"]=df_forecast["Date"].dt.month
    y_last=df_forecast["Year"].max(); y_prev=y_last-1
    s_last=df_forecast[df_forecast["Year"]==y_last]["Sales"].sum()
    s_prev=df_forecast[df_forecast["Year"]==y_prev]["Sales"].sum()
    yoy=(s_last-s_prev)/(s_prev+1e-9)*100 if s_prev>0 else 0
    yoy_sign="🔺" if yoy>0 else "🔻"
    period_str=(f"{df_monthly['Date'].min().strftime('%b %Y')} → {df_monthly['Date'].max().strftime('%b %Y')}")
    for col,label,val,sub in [
        (k1,"PERIOD",period_str,f"{len(df_monthly)} months"),
        (k2,"AVG MONTHLY",f"₹{avg_cr:.1f} Cr","average"),
        (k3,"PEAK MONTH",f"₹{peak_cr:.1f} Cr",peak_m),
        (k4,"TOTAL",f"₹{total_cr:.1f} Cr","all months"),
        (k5,"YoY GROWTH",f"{yoy_sign}{abs(yoy):.1f}%",f"{y_prev}→{y_last}"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:{'18px' if len(val)>8 else '24px'};">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if outlier_summary["n_outliers"]>0:
        badge_cls="clean-badge" if remove_outliers else "outlier-badge"
        badge_msg=(f'✅ {outlier_summary["n_outliers"]} outlier(s) detected and cleaned automatically' if remove_outliers else f'⚠️ {outlier_summary["n_outliers"]} outlier(s) detected (cleaning disabled)')
        st.markdown(f'<span class="{badge_cls}">{badge_msg}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="clean-badge">✅ No outliers detected — data is clean</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_col,_=st.columns([1,3])
    with run_col: run_btn=st.button("🚀 Run Forecast",use_container_width=True)

    if "forecast_done" not in st.session_state: st.session_state.forecast_done=False
    if run_btn:
        st.session_state.forecast_done=True
        st.session_state.forecast_key=f"{filter_label}_{remove_outliers}_{forecast_periods}"

    if not st.session_state.forecast_done:
        st.info("👆 Click **Run Forecast** to generate predictions.")
        st.plotly_chart(plot_outliers(raw_sales,cleaned_sales,outlier_flags,df_monthly["Date"]),use_container_width=True)
        return

    # ── Training ──────────────────────────────────────────────────────────────
    y=df_forecast["Sales"].values
    with st.spinner("🔧 Training models + ensemble..."):
        prog=st.progress(0)
        models,ens=run_models(y, season_len=dls["season_len"], forecast_periods=forecast_periods)
        prog.progress(100)

    last_date=df_forecast["Date"].max()
    future_dates=pd.date_range(start=last_date+pd.offsets.MonthBegin(1),periods=forecast_periods,freq="MS")
    residuals=np.array(y)-np.array(ens["fitted"])
    lo,hi_ci=confidence_intervals(np.array(ens["forecast"]),residuals)
    all_models=models+[ens]
    metrics_df=pd.DataFrame([compute_metrics(y,m["fitted"],m["name"]) for m in all_models])

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1,tab2,tab3,tab4=st.tabs(["🔮 Forecast","📊 Data Quality","🎯 Model Accuracy","📋 Metrics Table"])

    with tab1:
        st.markdown("<div class='section-header'>🔮 6-Month Forecast Results</div>", unsafe_allow_html=True)
        st.markdown(accuracy_tier_badge(ens["mape"]), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        fc_arr=np.array(ens["forecast"]); total_fc=fc_arr.sum()/1e7
        best_month_idx=np.argmax(fc_arr); best_month_val=fc_arr[best_month_idx]/1e7
        last_actual=y[-1]; growth=(fc_arr[0]-last_actual)/(last_actual+1e-9)*100
        ka,kb,kc,kd=st.columns(4)
        for col,label,val,sub in [
            (ka,"6M FORECAST TOTAL",f"₹{total_fc:.1f} Cr","ensemble"),
            (kb,"MODEL ACCURACY",f"{ens['mape']:.1f}%","ensemble MAPE"),
            (kc,"PEAK FORECAST MONTH",f"₹{best_month_val:.1f} Cr",future_dates[best_month_idx].strftime("%b %Y")),
            (kd,"NEXT MONTH CHANGE",f"{'+' if growth>0 else ''}{growth:.1f}%","vs last actual"),
        ]:
            with col:
                color="#50d090" if label=="MODEL ACCURACY" and ens["mape"]<10 else "#a09cf7"
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color};">{val}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(plot_main_forecast(df_forecast,models,ens,future_dates,lo,hi_ci),use_container_width=True)
        st.plotly_chart(plot_detailed_forecast(df_forecast,ens,future_dates,lo,hi_ci),use_container_width=True)
        st.markdown("<div class='section-header'>📋 Forecast Table</div>", unsafe_allow_html=True)
        fc_table=[]
        for i,(fd,fv,fl,fh_v) in enumerate(zip(future_dates,fc_arr,lo,hi_ci)):
            prev=fc_arr[i-1] if i>0 else last_actual; mom=(fv-prev)/(prev+1e-9)*100
            fc_table.append({"Month":fd.strftime("%b %Y"),"Forecast (₹ Cr)":f"₹{fv/1e7:.2f} Cr","Lower CI (₹ Cr)":f"₹{fl/1e7:.1f} Cr","Upper CI (₹ Cr)":f"₹{fh_v/1e7:.1f} Cr","MoM Change":f"{'🔺' if mom>0 else '🔻'} {mom:+.1f}%"})
        st.dataframe(pd.DataFrame(fc_table),use_container_width=True,hide_index=True)
        excel_bytes=build_excel(df_forecast,ens,models,future_dates,metrics_df,lo,hi_ci,filter_label)
        st.download_button("⬇️ Download Full Results (Excel)",data=excel_bytes,file_name=f"forecast_{filter_label.replace(' ','_')}.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)

    with tab2:
        st.markdown("<div class='section-header'>🔍 Data Quality & Outlier Analysis</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_outliers(raw_sales,cleaned_sales,outlier_flags,df_monthly["Date"]),use_container_width=True)
        st.plotly_chart(plot_seasonality_heatmap(df_forecast),use_container_width=True)
        if outlier_summary["n_outliers"]>0:
            st.markdown("**Outlier Details:**")
            ol_rows=[{"Month":df_monthly.iloc[idx]["Date"].strftime("%b %Y"),"Raw (₹ Cr)":f"₹{raw_sales.iloc[idx]/1e7:.2f} Cr","Cleaned (₹ Cr)":f"₹{cleaned_sales.iloc[idx]/1e7:.2f} Cr"} for idx in outlier_summary["outlier_indices"] if idx<len(df_monthly)]
            st.dataframe(pd.DataFrame(ol_rows),use_container_width=True,hide_index=True)

    with tab3:
        st.plotly_chart(plot_model_accuracy(all_models),use_container_width=True)
        st.markdown("**Model Weight in Ensemble (higher weight = better model):**")
        ens_weights=ens["params"].get("weights",[])
        weight_data=[{"Model":m["name"],"MAPE %":f"{m['mape']:.2f}%","Weight":f"{w:.3f}  ({w*100:.1f}%)","Contribution":"⭐⭐⭐" if w==max(ens_weights) else ("⭐⭐" if w>0.25 else "⭐")} for m,w in zip(models,ens_weights)]
        st.dataframe(pd.DataFrame(weight_data),use_container_width=True,hide_index=True)

    with tab4:
        st.markdown("<div class='section-header'>📋 Full Model Performance Metrics</div>", unsafe_allow_html=True)
        if dls["tier"]!="full":
            st.markdown(f"""<div class="data-warning-banner">
                <div class="banner-title">⚠️ Accuracy Limited by Data Length</div>
                <div class="banner-body">Current data: <b>{len(df_monthly)} months</b> — upload <b>{MIN_MONTHS_FULL_ACCURACY-len(df_monthly)} more months</b> to reach Full Accuracy Mode.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="data-good-banner">
                <div class="banner-title">✅ Full Accuracy Mode Active</div>
                <div class="banner-body">{len(df_monthly)} months loaded — all seasonal patterns captured. Models running at maximum precision.</div>
            </div>""", unsafe_allow_html=True)
        st.dataframe(metrics_df.style.format({"MAE (₹ Cr)":"{:.2f}","RMSE (₹ Cr)":"{:.2f}","MAPE (%)":"{:.2f}","R²":"{:.4f}"}).background_gradient(subset=["MAPE (%)"],cmap="RdYlGn_r"),use_container_width=True,hide_index=True)
        st.markdown("""
        **Metric Guide:**
        - **MAPE** — Mean Absolute Percentage Error. Lower = better. <5% = Excellent, 5–10% = Good, >15% = Poor.
        - **R²** — Coefficient of determination. Closer to 1.0 = better fit.
        - **MAE / RMSE** — Error in ₹ Crore (lower = better).
        """)


if __name__ == "__main__":
    main()
