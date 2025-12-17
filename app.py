import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------
# 0. Global Config
# --------------------------
st.set_page_config(page_title="Energy Surrogate MVP", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #2ecc71; }
    .big-number { font-size: 24px; font-weight: bold; color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = r"C:\EnergyML\Dataset.csv"
KWH_TO_GJ = 0.0036
END_USE_COLS = ["EUI (Total) (kWh/m²)", "EUI (Electricity) (kWh/m²)", "EUI (Gas) (kWh/m²)",
                "Space heating (gas) (kWh/m²)", "Space heating (elec) (kWh/m²)"]

# --------------------------
# 1. Hybrid Model Class (Tuned)
# --------------------------
class PhysicsHybridModel:
    def __init__(self):
        self.lin_model = LinearRegression()
        # Tuned: Lower depth & learning rate to let Physics (Linear) dominate
        self.xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4)

    def fit(self, X, y):
        self.lin_model.fit(X, y)
        linear_pred = self.lin_model.predict(X)
        residuals = y - linear_pred
        self.xgb_model.fit(X, residuals)
        return self

    def predict(self, X):
        return self.lin_model.predict(X) + self.xgb_model.predict(X)

# --------------------------
# 2. Data Pipeline
# --------------------------
def parse_hvac_strict(s: str):
    s_lower = str(s).lower()
    if "furnace" in s_lower: sys, fuel, vent = "Furnace", "Gas", "Unknown"
    elif "ahu" in s_lower: sys, fuel, vent = "AHU", "Electric", "CV"
    else: sys, fuel, vent = "Other", "Unknown", "Unknown"
    has_cooling = 1
    return pd.Series({"System_Type": sys, "Heating_Fuel": fuel, "Has_Cooling": has_cooling, "Ventilation_Mode": vent})

@st.cache_data(show_spinner=False)
def load_and_engineer_data():
    df = None
    for enc in ['utf-8', 'cp1252', 'latin1']:
        try: df = pd.read_csv(DATA_PATH, encoding=enc); break
        except: continue
    if df is None:
        try: df = pd.read_excel(DATA_PATH.replace('.csv', '.xlsx'))
        except: st.error("Load failed."); st.stop()

    hvac_col = next((c for c in df.columns if "ApacheHVAC" in c), "ApacheHVAC File (ASP)")
    if hvac_col in df.columns:
        hvac_features = df[hvac_col].apply(parse_hvac_strict)
    else: st.error("Missing HVAC column"); st.stop()

    u_map = {
        "External Wall Construction U Values (W/m2.K)": "Wall_U_Value",
        "External Window Construction U Values (W/m2.K)": "Window_U_Value",
        "Roof Construction U Values (W/m2.K)": "Roof_U_Value"
    }
    for t, n in u_map.items():
        if t not in df.columns:
            match = [c for c in df.columns if n.split("_")[0] in c and "U Value" in c]
            if match: df.rename(columns={match[0]: t}, inplace=True)
            
    for col in u_map.keys():
        if col not in df.columns:
            r_src = "External Wall Construction.1" if "Wall" in col else ("External Window Construction" if "Window" in col else "Roof Construction")
            if r_src in df.columns:
                df[col] = df[r_src].astype(str).apply(lambda x: 1.0/float(re.search(r"R-?(\d+(\.\d+)?)", x).group(1)) if re.search(r"R-?(\d+(\.\d+)?)", x) else 0.5)

    df = df.rename(columns=u_map)
    clim_map = {"HDD (Base 18C)": "HDD (Base 18C)", "CDD (Base 18C)": "CDD (Base 18C)"}
    if "Avg Humidity (%)" in df.columns: clim_map["Avg Humidity (%)"] = "Avg Humidity (%)"
    elif "Average Relative Humidity (%)" in df.columns: clim_map["Average Relative Humidity (%)"] = "Avg Humidity (%)"
    df = df.rename(columns=clim_map)

    req = ["Wall_U_Value", "Window_U_Value", "Roof_U_Value", "HDD (Base 18C)", "CDD (Base 18C)", "Avg Humidity (%)"]
    valid = [c for c in req if c in df.columns]
    X_raw = pd.concat([hvac_features, df[valid]], axis=1)
    mask = ~X_raw.isna().any(axis=1)
    X_raw = X_raw.loc[mask].reset_index(drop=True)
    df_clean = df.loc[mask].reset_index(drop=True)

    X = pd.get_dummies(X_raw, columns=["System_Type", "Heating_Fuel", "Ventilation_Mode"], drop_first=False)
    for c in X.columns: 
        if X[c].dtype == bool: X[c] = X[c].astype(int)

    return df_clean, X_raw, X

@st.cache_resource(show_spinner=False)
def train_hybrid_models():
    with st.spinner("🚀 Training Hybrid Models..."):
        df_clean, X_raw, X = load_and_engineer_data()
        X_train, X_test, y_idx_train, y_idx_test = train_test_split(X, df_clean.index, test_size=0.2, random_state=42)
        X_raw_test = X_raw.loc[y_idx_test]
        
        models = {}
        val_data = {"X_test": X_test, "X_raw_test": X_raw_test, "y_test": {}}
        
        targets = [t for t in END_USE_COLS if t in df_clean.columns]
        for col in targets:
            y = df_clean[col]
            m = PhysicsHybridModel() 
            m.fit(X_train, y.loc[y_idx_train])
            models[col] = m
            val_data["y_test"][col] = y.loc[y_idx_test]
            
    return models, X_train.columns, val_data, X_raw

def build_input(wall, win, roof, hdd, cdd, hum, mode, cols):
    row = {c: 0.0 for c in cols}
    row["Wall_U_Value"] = 1.0/wall
    row["Window_U_Value"] = 1.0/win
    row["Roof_U_Value"] = 1.0/roof
    row["HDD (Base 18C)"] = hdd
    row["CDD (Base 18C)"] = cdd
    row["Avg Humidity (%)"] = hum
    row["Has_Cooling"] = 1
    
    if mode == "Gas Furnace System": sys, fuel, vent = "Furnace", "Gas", "Unknown"
    else: sys, fuel, vent = "AHU", "Electric", "CV"
    
    for k, v in [("System_Type", sys), ("Heating_Fuel", fuel), ("Ventilation_Mode", vent)]:
        if f"{k}_{v}" in row: row[f"{k}_{v}"] = 1.0
    return pd.DataFrame([row])

# --------------------------
# 3. UI
# --------------------------
models, feat_cols, val_data, X_raw_full = train_hybrid_models()

with st.sidebar:
    st.header("🎛️ Hybrid Design Inputs")
    with st.expander("🏗️ Envelope (R-Values)", expanded=True):
        wall = st.number_input("Wall R", 1.0, 100.0, 20.0, help="Impacts Heating")
        win = st.number_input("Window R", 0.5, 20.0, 3.0, help="High Impact on Energy")
        roof = st.number_input("Roof R", 1.0, 150.0, 40.0, help="Low Impact (Small Area)")
    
    sys = st.radio("System", ["Gas Furnace System", "Electric AHU System"])
    
    with st.expander("🌍 Climate"):
        hdd = st.number_input("HDD", 0, 10000, 5000)
        cdd = st.number_input("CDD", 0, 3000, 150)
        hum = st.slider("Humidity", 0.0, 1.0, 0.6)
    
    st.markdown("---")
    st.caption("Hybrid Model: Linear (Physics) + XGB (Residuals)")

st.title("⚡ Hybrid Energy Surrogate")

tab1, tab2 = st.tabs(["🔮 Prediction", "🧪 Hybrid Validation"])

with tab1:
    X_in = build_input(wall, win, roof, hdd, cdd, hum, sys, feat_cols)
    preds = {k: m.predict(X_in)[0] for k, m in models.items()}
    
    e_tot = preds.get("EUI (Total) (kWh/m²)", 0) * KWH_TO_GJ
    e_gas = preds.get("EUI (Gas) (kWh/m²)", 0) * KWH_TO_GJ
    e_elec = preds.get("EUI (Electricity) (kWh/m²)", 0) * KWH_TO_GJ
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total EUI", f"{e_tot:.3f} GJ/m²", delta="Model Output")
    c2.metric("Gas", f"{e_gas:.3f} GJ/m²")
    c3.metric("Electricity", f"{e_elec:.3f} GJ/m²")
    
    st.info("💡 **Tip:** If Roof R-value seems to have no effect, it is because roofs are a small % of total building area in MURBs compared to Walls & Windows.")

with tab2:
    st.subheader("Validation (20% Hidden Data)")
    target = st.selectbox("Select Target", list(models.keys()))
    
    y_true = val_data["y_test"][target]
    y_pred = models[target].predict(val_data["X_test"])
    
    r2 = r2_score(y_true, y_pred)
    
    st.metric("Hybrid R² Score", f"{r2:.4f}")
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15, color='#2ecc71')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=1)
    ax.set_xlabel("Actual Simulation")
    ax.set_ylabel("Hybrid Prediction")
    ax.set_title("Prediction Accuracy")
    st.pyplot(fig)