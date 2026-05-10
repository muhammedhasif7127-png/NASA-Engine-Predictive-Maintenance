import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NASA Engine Health Monitor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
    }
    .stApp {
        background-color: #0a0e1a;
        color: #e0e8ff;
    }
    .metric-card {
        background: linear-gradient(135deg, #0f1628 0%, #1a2444 100%);
        border: 1px solid #2a3a6a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .rul-value {
        font-family: 'Share Tech Mono', monospace;
        font-size: 56px;
        font-weight: bold;
        line-height: 1;
    }
    .status-excellent { color: #00ff9d; }
    .status-warning   { color: #ffb700; }
    .status-critical  { color: #ff3860; }
    .sensor-label {
        font-size: 12px;
        color: #8899bb;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    h1, h2, h3 { font-family: 'Rajdhani', sans-serif; font-weight: 700; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Rajdhani', sans-serif;
        font-size: 16px;
        font-weight: 600;
    }
    div[data-testid="stSidebarContent"] {
        background: #080c18;
        border-right: 1px solid #1a2444;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    paths = [
        'nasa_engine_rul_model.pkl',
        'models/nasa_engine_rul_model.pkl',
        'model.pkl'
    ]
    for p in paths:
        if os.path.exists(p):
            return joblib.load(p), True
    return None, False

model, model_loaded = load_model()

# ─── Feature definitions (C-MAPSS sensors used in training) ───
SENSOR_INFO = {
    's2':  ('Fan Inlet Temp (°R)',       550.0, 650.0, 600.0,  1.0),
    's3':  ('HPC Outlet Temp (°R)',     1550.0,1650.0,1600.0,  5.0),
    's4':  ('LPT Outlet Temp (°R)',     1370.0,1430.0,1400.0,  5.0),
    's6':  ('Total Pressure (psia)',      20.0,  25.0,  21.0,  0.1),
    's7':  ('Bypass Ratio',              13.0,  15.0,  14.0,  0.1),
    's8':  ('Bleed Enthalpy',           390.0, 400.0, 395.0,  0.5),
    's9':  ('HPT coolant bleed (%)',     9000., 9500., 9200.,  10.),
    's11': ('Static Pressure (psia)',     46.0,  52.0,  47.0,  0.1),
    's12': ('Fuel Flow Ratio',           520.0, 540.0, 530.0,  0.5),
    's13': ('Corrected Core Speed',      2380., 2400., 2390.,  1.0),
    's14': ('Bypass Ratio (alt)',        8000., 8500., 8200., 10.0),
    's15': ('Bleed Enthalpy (alt)',        8.0,  10.0,   8.5,  0.1),
    's17': ('Vibration Level',           390.0, 400.0, 395.0,  0.5),
    's20': ('HPT efficiency',             38.0,  42.0,  39.0,  0.1),
    's21': ('LPT efficiency',             23.0,  24.0,  23.5,  0.1),
}

# ─── Header ────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("## ✈️")
with col_title:
    st.markdown("# NASA AIRCRAFT ENGINE HEALTH MONITOR")
    st.markdown("<span style='color:#8899bb;font-size:14px;'>C-MAPSS Predictive Maintenance · Gradient Boosting RUL Predictor · RMSE ≈ 33.7 cycles</span>", unsafe_allow_html=True)

st.markdown("---")

if not model_loaded:
    st.warning("⚠️ **Model file not found** (`nasa_engine_rul_model.pkl`). Running in **demo mode** with physics-based estimation. Place your trained `.pkl` file in the same directory to enable ML predictions.")

# ─── Sidebar: Sensor Inputs ────────────────────────────────────
st.sidebar.markdown("## 📡 Live Sensor Panel")
st.sidebar.markdown("<span style='color:#8899bb;font-size:12px;'>Adjust sliders to simulate engine sensor readings</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")

sensor_vals = {}
for key, (label, lo, hi, default, step) in SENSOR_INFO.items():
    sensor_vals[key] = st.sidebar.slider(
        f"{key.upper()} — {label}",
        min_value=float(lo),
        max_value=float(hi),
        value=float(default),
        step=float(step)
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**Operational Settings**")
op_setting_1 = st.sidebar.selectbox("Flight Condition", [0, 1, 2, 3, 4, 5, 6], index=0)
cycles_so_far = st.sidebar.number_input("Cycles Elapsed", min_value=1, max_value=300, value=50)

analyze = st.sidebar.button("🚀 ANALYZE ENGINE HEALTH", use_container_width=True)

# ─── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Health Dashboard", "📈 Sensor Analysis", "📁 Batch Prediction"])

# ─── Prediction Logic ───────────────────────────────────────────
def predict_rul(sensor_vals, op_setting_1, cycles_so_far):
    """Use model if loaded, else use physics-based proxy."""
    features = list(sensor_vals.values()) + [op_setting_1, cycles_so_far]
    features_arr = np.array(features).reshape(1, -1)

    if model_loaded:
        try:
            rul = float(model.predict(features_arr)[0])
            return max(0, rul), "model"
        except Exception:
            pass

    # Physics-based fallback (same logic as original, but improved)
    s2, s11, s9 = sensor_vals['s2'], sensor_vals['s11'], sensor_vals['s9']
    wear = (s2 - 550) * 2 + (s11 - 46) * 15 + (s9 - 9000) * 0.01
    wear += cycles_so_far * 0.4
    rul = max(0, 200 - wear)
    return rul, "demo"

def get_status(rul):
    if rul > 100:  return "EXCELLENT", "status-excellent", "#00ff9d", 0.85
    if rul > 50:   return "WARNING",   "status-warning",   "#ffb700", 0.50
    return              "CRITICAL",  "status-critical",  "#ff3860", 0.15

# ─── Tab 1: Dashboard ──────────────────────────────────────────
with tab1:
    if analyze or True:  # always show default state
        rul, mode = predict_rul(sensor_vals, op_setting_1, cycles_so_far)
        status_text, status_class, status_color, gauge_frac = get_status(rul)

        # Top metrics row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='sensor-label'>Remaining Useful Life</div>
                <div class='rul-value {status_class}'>{rul:.0f}</div>
                <div style='color:#8899bb;font-size:13px;margin-top:4px;'>cycles</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='sensor-label'>Engine Status</div>
                <div class='rul-value {status_class}' style='font-size:32px;margin-top:8px;'>{status_text}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            health_pct = min(100, int(rul / 200 * 100))
            st.markdown(f"""
            <div class='metric-card'>
                <div class='sensor-label'>Health Score</div>
                <div class='rul-value' style='color:#60a0ff;'>{health_pct}%</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            mode_label = "🤖 ML Model" if mode == "model" else "⚙️ Demo Mode"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='sensor-label'>Prediction Engine</div>
                <div style='font-size:20px;margin-top:12px;font-weight:600;color:#e0e8ff;'>{mode_label}</div>
                <div style='color:#8899bb;font-size:12px;'>Cycles elapsed: {cycles_so_far}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge chart + recommendation
        col_gauge, col_rec = st.columns([1, 1])
        with col_gauge:
            fig, ax = plt.subplots(figsize=(6, 3.5), subplot_kw={'aspect': 'equal'})
            fig.patch.set_facecolor('#0a0e1a')
            ax.set_facecolor('#0a0e1a')

            # Draw gauge arc background
            theta = np.linspace(np.pi, 0, 200)
            for i, (t1, t2, color) in enumerate(zip(
                [np.pi, np.pi*2/3, np.pi/3],
                [np.pi*2/3, np.pi/3, 0],
                ['#ff3860', '#ffb700', '#00ff9d']
            )):
                seg = np.linspace(t1, t2, 50)
                ax.plot(np.cos(seg), np.sin(seg), color=color, linewidth=18, alpha=0.25)

            # Needle
            needle_angle = np.pi * (1 - gauge_frac)
            ax.annotate('', xy=(0.65*np.cos(needle_angle), 0.65*np.sin(needle_angle)),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=status_color, lw=3))
            ax.add_patch(plt.Circle((0, 0), 0.06, color=status_color, zorder=5))

            # Labels
            for val, angle in [(0, np.pi), (100, np.pi/2), (200, 0)]:
                ax.text(1.15*np.cos(angle), 1.15*np.sin(angle), str(val),
                       ha='center', va='center', color='#8899bb', fontsize=10)

            ax.text(0, -0.25, f"{rul:.0f} cycles", ha='center', va='center',
                   color=status_color, fontsize=18, fontweight='bold',
                   fontfamily='monospace')

            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-0.4, 1.3)
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

        with col_rec:
            st.markdown("### 🔧 Maintenance Recommendation")
            if rul > 100:
                st.success(f"""
**✅ Engine health is EXCELLENT**

- No immediate action required
- Schedule next inspection at **{int(rul*0.7):.0f} cycles**
- Continue standard monitoring protocol
- Key sensors nominal: S9, S11
                """)
            elif rul > 50:
                st.warning(f"""
**⚠️ Engine showing early wear signs**

- Schedule maintenance within **{int(rul*0.5):.0f} cycles**
- Inspect Sensor 11 (Static Pressure) closely
- Run diagnostics on HPT coolant bleed (S9)
- Increase monitoring frequency
                """)
            else:
                st.error(f"""
**🚨 CRITICAL — Immediate attention required**

- **Ground aircraft if RUL < 10 cycles**
- Emergency inspection recommended NOW
- Prioritize: S11, S9, S2 sensor checks
- Estimated failure within **{rul:.0f} cycles**
                """)

# ─── Tab 2: Sensor Analysis ────────────────────────────────────
with tab2:
    st.markdown("### Sensor Reading vs. Healthy Baseline")
    st.markdown("<span style='color:#8899bb'>Green = within normal range. Red = degraded. Bar height shows deviation from baseline.</span>", unsafe_allow_html=True)

    # Compute % deviation from midpoint
    deviations = {}
    for key, (label, lo, hi, default, _) in SENSOR_INFO.items():
        midpoint = (lo + hi) / 2
        val = sensor_vals[key]
        dev_pct = abs(val - default) / (hi - lo) * 100
        deviations[key] = (label, dev_pct, val, default)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    fig2.patch.set_facecolor('#0a0e1a')
    ax2.set_facecolor('#0f1628')

    keys = list(deviations.keys())
    devs = [deviations[k][1] for k in keys]
    colors = ['#ff3860' if d > 25 else '#ffb700' if d > 10 else '#00ff9d' for d in devs]

    bars = ax2.bar(keys, devs, color=colors, edgecolor='#1a2444', linewidth=0.8, width=0.6)
    ax2.axhline(10, color='#ffb700', linestyle='--', alpha=0.5, linewidth=1, label='Caution threshold (10%)')
    ax2.axhline(25, color='#ff3860', linestyle='--', alpha=0.5, linewidth=1, label='Critical threshold (25%)')

    ax2.set_ylabel('Deviation from Baseline (%)', color='#8899bb', fontsize=11)
    ax2.set_xlabel('Sensor', color='#8899bb', fontsize=11)
    ax2.tick_params(colors='#8899bb')
    ax2.spines['bottom'].set_color('#2a3a6a')
    ax2.spines['left'].set_color('#2a3a6a')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_facecolor('#0f1628')
    legend = ax2.legend(facecolor='#0f1628', edgecolor='#2a3a6a', labelcolor='#8899bb')
    st.pyplot(fig2)
    plt.close()

    # Sensor table
    st.markdown("### 📋 Sensor Readings Table")
    df_sensors = pd.DataFrame([
        {
            "Sensor": k.upper(),
            "Description": SENSOR_INFO[k][0],
            "Current Value": f"{sensor_vals[k]:.2f}",
            "Baseline": f"{SENSOR_INFO[k][3]:.2f}",
            "Range": f"{SENSOR_INFO[k][1]:.1f} – {SENSOR_INFO[k][2]:.1f}",
            "Status": "🔴 Degraded" if deviations[k][1] > 25 else "🟡 Caution" if deviations[k][1] > 10 else "🟢 Normal"
        }
        for k in keys
    ])
    st.dataframe(df_sensors, use_container_width=True, hide_index=True)

# ─── Tab 3: Batch Prediction ───────────────────────────────────
with tab3:
    st.markdown("### 📁 Batch Engine Prediction")
    st.markdown("Upload a CSV with sensor columns to predict RUL for multiple engines at once.")

    st.markdown("**Required columns:** `s2, s3, s4, s6, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21, op_setting_1, cycles`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df_batch = pd.read_csv(uploaded)
            st.dataframe(df_batch.head(), use_container_width=True)

            sensor_cols = list(SENSOR_INFO.keys()) + ['op_setting_1', 'cycles']
            missing = [c for c in sensor_cols if c not in df_batch.columns]

            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                X = df_batch[sensor_cols].values
                if model_loaded:
                    preds = model.predict(X)
                else:
                    # demo fallback per row
                    preds = []
                    for row in X:
                        sv = {k: row[i] for i, k in enumerate(SENSOR_INFO.keys())}
                        rul_p, _ = predict_rul(sv, row[-2], row[-1])
                        preds.append(rul_p)
                    preds = np.array(preds)

                df_batch['Predicted_RUL'] = np.maximum(0, preds).astype(int)
                df_batch['Status'] = df_batch['Predicted_RUL'].apply(
                    lambda x: '🟢 EXCELLENT' if x > 100 else '🟡 WARNING' if x > 50 else '🔴 CRITICAL'
                )
                st.success(f"✅ Predictions complete for {len(df_batch)} engines.")
                st.dataframe(df_batch[['Predicted_RUL', 'Status']].join(df_batch.drop(columns=['Predicted_RUL','Status'])), use_container_width=True)

                csv_out = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results CSV", csv_out, "rul_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("👆 Upload a CSV file to begin batch prediction.")

# ─── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#3a4a7a;font-size:13px;font-family:Share Tech Mono,monospace;'>"
    "NASA C-MAPSS Dataset · Gradient Boosting Regressor · RMSE ≈ 33.72 · "
    "Developed by <b style='color:#5a7aaa'>Mohammed Hasif</b></p>",
    unsafe_allow_html=True
)
