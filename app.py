import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------
# 0. Global Config & Styling
# --------------------------
st.set_page_config(
    page_title="Energy Surrogate MVP", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4c72b0;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
    }
    .big-number {
        font-size: 24px; 
        font-weight: bold; 
        color: #2c3e50;
    }
    .unit-label {
        font-size: 14px; 
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = r"C:\EnergyML\Dataset.csv"

# Conversion Factors
KWH_TO_GJ = 0.0036

END_USE_COLS = [
    "EUI (Total) (kWh/m²)",
    "EUI (Electricity) (kWh/m²)",
    "EUI (Gas) (kWh/m²)",
    "Interior lighting (kWh/m²)",
    "Space heating (gas) (kWh/m²)",
    "Space heating (elec) (kWh/m²)",
    "Space cooling (kWh/m²)",
    "Pumps (kWh/m²)",
    "Fans interior (kWh/m²)",
    "Heat Rejection (kWh/m²)",
    "DHW heating (gas) (kWh/m²)",
    "DHW heating (elec) (kWh/m²)",
    "Receptacle equipment (kWh/m²)",
    "Elevators escalators (kWh/m²)",
]

# --------------------------
# 1. Robust Data Pipeline
# --------------------------

def parse_hvac_strict(s: str):
    s_lower = str(s).lower()
    
    # Logic matching your dataset's specific two strings
    if "furnace" in s_lower: 
        system_type = "Furnace"
        heating_fuel = "Gas"
        vent = "Unknown" # The furnace string didn't specify CV/VAV
    elif "ahu" in s_lower: 
        system_type = "AHU"
        heating_fuel = "Electric"
        vent = "CV"
    else: 
        system_type = "Other"
        heating_fuel = "Unknown"
        vent = "Unknown"

    has_cooling = 1 if ("electric cool" in s_lower or "ac" in s_lower or "cool" in s_lower) else 0
    
    return pd.Series({
        "System_Type": system_type,
        "Heating_Fuel": heating_fuel,
        "Has_Cooling": has_cooling,
        "Ventilation_Mode": vent
    })

@st.cache_data(show_spinner=False)
def load_and_engineer_data():
    df = None
    for encoding in ['utf-8', 'cp1252', 'latin1']:
        try:
            df = pd.read_csv(DATA_PATH, encoding=encoding)
            break
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    if df is None:
        try:
            df = pd.read_excel(DATA_PATH.replace('.csv', '.xlsx'))
        except Exception:
            st.error(f"❌ Could not load data from {DATA_PATH}.")
            st.stop()

    hvac_col = "ApacheHVAC File (ASP)"
    if hvac_col not in df.columns:
        match = next((c for c in df.columns if "ApacheHVAC" in c), None)
        if match: df.rename(columns={match: hvac_col}, inplace=True)
        else: st.error("Missing 'ApacheHVAC' column."); st.stop()
        
    hvac_features = df[hvac_col].apply(parse_hvac_strict)

    u_map = {
        "External Wall Construction U Values (W/m2.K)": "Wall_U_Value",
        "External Window Construction U Values (W/m2.K)": "Window_U_Value",
        "Roof Construction U Values (W/m2.K)": "Roof_U_Value"
    }
    for target, name in u_map.items():
        if target not in df.columns:
            partial = [c for c in df.columns if name.split("_")[0] in c and "U Value" in c]
            if partial: df.rename(columns={partial[0]: target}, inplace=True)
            
    for col in u_map.keys():
        if col not in df.columns:
             r_source = "External Wall Construction.1" if "Wall" in col else ("External Window Construction" if "Window" in col else "Roof Construction")
             if r_source in df.columns:
                 df[col] = df[r_source].astype(str).apply(lambda x: 1.0/float(re.search(r"R-?(\d+(\.\d+)?)", x).group(1)) if re.search(r"R-?(\d+(\.\d+)?)", x) else 0.5)

    df = df.rename(columns=u_map)

    clim_map = {"HDD (Base 18C)": "HDD (Base 18C)", "CDD (Base 18C)": "CDD (Base 18C)"}
    if "Average Relative Humidity (%)" in df.columns: clim_map["Average Relative Humidity (%)"] = "Avg Humidity (%)"
    elif "Avg Humidity (%)" in df.columns: clim_map["Avg Humidity (%)"] = "Avg Humidity (%)"
    df = df.rename(columns=clim_map)

    req_cols = ["Wall_U_Value", "Window_U_Value", "Roof_U_Value", "HDD (Base 18C)", "CDD (Base 18C)", "Avg Humidity (%)"]
    valid_cols = [c for c in req_cols if c in df.columns]
    
    X_raw = pd.concat([hvac_features, df[valid_cols]], axis=1)
    mask = ~X_raw.isna().any(axis=1)
    X_raw = X_raw.loc[mask].reset_index(drop=True)
    df_clean = df.loc[mask].reset_index(drop=True)

    X = pd.get_dummies(X_raw, columns=["System_Type", "Heating_Fuel", "Ventilation_Mode"], drop_first=False)
    for c in X.columns:
        if X[c].dtype == bool: X[c] = X[c].astype(int)

    return df_clean, X_raw, X

@st.cache_resource(show_spinner=False)
def train_surrogate_models():
    with st.spinner("🚀 Training Business Logic Models..."):
        df_clean, X_raw, X = load_and_engineer_data()
        
        # 80/20 Split for honest validation
        X_train, X_test, y_indices_train, y_indices_test = train_test_split(
            X, df_clean.index, test_size=0.2, random_state=42
        )
        
        X_raw_test = X_raw.loc[y_indices_test]

        bounds = {}
        if "Wall_U_Value" in X_raw.columns:
            r_vals = 1.0 / X_raw["Wall_U_Value"].replace(0, 0.001)
            bounds["Wall_R"] = (r_vals.min(), r_vals.max())
        if "Roof_U_Value" in X_raw.columns:
            r_vals = 1.0 / X_raw["Roof_U_Value"].replace(0, 0.001)
            bounds["Roof_R"] = (r_vals.min(), r_vals.max())
            
        targets = [c for c in END_USE_COLS if c in df_clean.columns]
        models = {}
        
        val_data = {"X_test": X_test, "X_raw_test": X_raw_test, "y_test": {}}

        for col in targets:
            y = df_clean[col]
            y_train = y.loc[y_indices_train]
            y_test = y.loc[y_indices_test]
            
            m = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, tree_method="hist")
            m.fit(X_train, y_train)
            models[col] = m
            val_data["y_test"][col] = y_test
            
    return models, X_train, X_raw, bounds, val_data

def build_input_row(wall_r, win_r, roof_r, hdd, cdd, hum, hvac_scenario, cols):
    row = {c: 0.0 for c in cols}
    row["Wall_U_Value"] = 1.0/wall_r if wall_r > 0 else 0
    row["Window_U_Value"] = 1.0/win_r if win_r > 0 else 0
    row["Roof_U_Value"] = 1.0/roof_r if roof_r > 0 else 0
    row["HDD (Base 18C)"] = hdd
    row["CDD (Base 18C)"] = cdd
    row["Avg Humidity (%)"] = hum
    
    # Couple the logic based on the two available strings
    if hvac_scenario == "Gas Furnace System":
        sys, fuel, vent = "Furnace", "Gas", "Unknown"
    else: # Electric AHU
        sys, fuel, vent = "AHU", "Electric", "CV"
        
    row["Has_Cooling"] = 1 # Both have cooling in your data
    
    for key, val in [("System_Type", sys), ("Heating_Fuel", fuel), ("Ventilation_Mode", vent)]:
        col_name = f"{key}_{val}"
        if col_name in row: row[col_name] = 1.0
            
    return pd.DataFrame([row])

# --------------------------
# 2. Main UI Logic
# --------------------------
models, X_train, X_raw, train_bounds, val_data = train_surrogate_models()
feat_cols = list(X_train.columns)

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("🎛️ Design Inputs")
    with st.expander("🏗️ Envelope (R-Values)", expanded=True):
        wall_r = st.number_input("Wall R-Value", 1.0, 100.0, 20.0)
        roof_r = st.number_input("Roof R-Value", 1.0, 150.0, 40.0)
        win_r = st.number_input("Window R-Value", 0.5, 20.0, 3.0)
        
        warnings = []
        if "Wall_R" in train_bounds and (wall_r < train_bounds["Wall_R"][0] or wall_r > train_bounds["Wall_R"][1]):
            warnings.append(f"⚠️ Wall R (Trained: {train_bounds['Wall_R'][0]:.1f}-{train_bounds['Wall_R'][1]:.1f})")

    with st.expander("⚙️ HVAC Scenario", expanded=True):
        # SIMPLIFIED TO TWO OPTIONS
        hvac_mode = st.radio("System Configuration", 
                             ["Gas Furnace System", "Electric AHU System"],
                             help="Select one of the two baseline systems available in the dataset.")

    with st.expander("🌍 Climate", expanded=False):
        hdd = st.number_input("HDD (18°C)", 0, 20000, 5000)
        cdd = st.number_input("CDD (18°C)", 0, 5000, 150)
        hum = st.slider("Avg Humidity", 0.0, 1.0, 0.60)
    
    st.markdown("---")
    st.markdown("### 💲 Business Assumptions")
    elec_cost = st.number_input("Elec Price ($/kWh)", 0.0, 1.0, 0.15)
    gas_cost = st.number_input("Gas Price ($/GJ)", 0.0, 50.0, 12.0)
    
    st.markdown("### ☁️ Carbon Factors")
    grid_intens = st.number_input("Grid (kgCO2/kWh)", 0.0, 1.0, 0.5)
    gas_intens = 50.0 # kgCO2/GJ approx standard

# --- MAIN PAGE ---
st.title("⚡ Energy Surrogate MVP")
st.markdown("### Financial & Carbon Impact Analysis")

if warnings:
    st.warning(" | ".join(warnings))

tab1, tab2 = st.tabs(["💰 Business Case", "🧪 Model Validation"])

# --- TAB 1: BUSINESS CASE ---
with tab1:
    X_input = build_input_row(wall_r, win_r, roof_r, hdd, cdd, hum, hvac_mode, feat_cols)
    
    # Get predictions in kWh (raw model unit)
    preds_kwh = {k: float(m.predict(X_input)[0]) for k, m in models.items()}
    
    eui_tot_kwh = preds_kwh.get("EUI (Total) (kWh/m²)", 0)
    eui_elec_kwh = preds_kwh.get("EUI (Electricity) (kWh/m²)", 0)
    eui_gas_kwh = preds_kwh.get("EUI (Gas) (kWh/m²)", 0)
    
    # CONVERSIONS
    eui_tot_gj = eui_tot_kwh * KWH_TO_GJ
    eui_gas_gj = eui_gas_kwh * KWH_TO_GJ
    # Electricity often stays in kWh for cost, but displayed in GJ
    eui_elec_gj = eui_elec_kwh * KWH_TO_GJ
    
    # COST CALC
    cost_elec = eui_elec_kwh * elec_cost
    cost_gas = eui_gas_gj * gas_cost
    cost_total = cost_elec + cost_gas
    
    # CARBON CALC
    carb_elec = eui_elec_kwh * grid_intens
    carb_gas = eui_gas_gj * gas_intens
    carb_total = carb_elec + carb_gas
    
    # --- METRICS ROW ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total EUI (GJ/m²)", f"{eui_tot_gj:.2f}", help=f"Equivalent to {eui_tot_kwh:.0f} kWh/m²")
    c2.metric("Op. Cost ($/m²)", f"${cost_total:.2f}", f"Gas: ${cost_gas:.2f} | Elec: ${cost_elec:.2f}")
    c3.metric("GHG Intensity", f"{carb_total:.1f}", "kgCO2e/m²")
    
    # Visual Indicator of Fuel Mix
    if eui_tot_gj > 0:
        gas_share = (eui_gas_gj / eui_tot_gj) * 100
        c4.metric("Gas Dependency", f"{gas_share:.0f}%", "of total energy")
    else:
        c4.metric("Gas Dependency", "0%", "")

    st.markdown("---")
    
    # --- CHARTS ---
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Energy Breakdown (GJ/m²)")
        # Filter end uses
        chart_data = {k: v * KWH_TO_GJ for k, v in preds_kwh.items() if "Total" not in k and "EUI (" not in k}
        chart_df = pd.DataFrame(list(chart_data.items()), columns=["End Use", "GJ/m²"]).sort_values("GJ/m²", ascending=True)
        
        # Interactive Chart
        st.bar_chart(chart_df.set_index("End Use"))
        
    with col_chart2:
        st.subheader("Cost Drivers ($/m²)")
        cost_df = pd.DataFrame([
            {"Source": "Electricity", "Cost": cost_elec},
            {"Source": "Natural Gas", "Cost": cost_gas}
        ])
        st.bar_chart(cost_df.set_index("Source"), color="#85bb65") # Dollar Green color

# --- TAB 2: VALIDATION ---
with tab2:
    st.subheader("🧪 Validation on Unseen Data (20% Test Set)")
    
    target_kwh = st.selectbox("Select Target Variable (kWh Model Units)", list(models.keys()))
    
    # Use HIDDEN TEST SET
    X_test = val_data["X_test"]
    y_true = val_data["y_test"][target_kwh]
    y_pred = models[target_kwh].predict(X_test)
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    k1, k2 = st.columns(2)
    k1.metric("R² Score", f"{r2:.4f}")
    k2.metric("RMSE", f"{rmse:.2f}", "kWh/m²")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Color by system type
    cats = val_data["X_raw_test"]["System_Type"]
    for c in cats.unique():
        mask = cats == c
        ax.scatter(y_true[mask], y_pred[mask], label=c, alpha=0.6, s=20)
    
    low, high = y_true.min(), y_true.max()
    ax.plot([low, high], [low, high], 'k--', lw=1, label="Perfect Prediction")
    ax.set_xlabel("Actual Simulation (kWh/m²)")
    ax.set_ylabel("AI Prediction (kWh/m²)")
    ax.legend()
    st.pyplot(fig)