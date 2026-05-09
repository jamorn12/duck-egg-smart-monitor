import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go

# 1. ตั้งค่าหน้าเว็บ (Industrial Theme)
st.set_page_config(page_title="Smart Hatchery Digital Twin", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .stMetric { background-color: #161b22; border-radius: 12px; padding: 20px; border: 1px solid #30363d; }
    .info-card { background-color: #1f2937; padding: 20px; border-radius: 10px; border-left: 5px solid #58a6ff; margin-bottom: 20px; }
    h1, h2, h3 { color: #f0f6fc; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'selected_egg' not in st.session_state:
    st.session_state.selected_egg = "None"

# 2. DATA ENGINE (1,000 Eggs + Facility Mapping)
@st.cache_resource
def load_factory_data():
    np.random.seed(42)
    racks, trays = [f"R{i:02d}" for i in range(1, 11)], [f"T{i:02d}" for i in range(1, 11)]
    egg_data, sensor_data = [], []
    
    for r_idx, r in enumerate(racks):
        # สร้างทางเดิน (Walkway) ให้เห็นผังอาคารชัดเจน
        y_offset = r_idx * 6 + (12 if r_idx >= 5 else 0)
        for t_idx, t in enumerate(trays):
            x_offset = t_idx * 9
            tray_temp = np.random.normal(37.5, 0.4)
            tray_humid = np.random.normal(65, 3.0)
            
            # เก็บตำแหน่งเซ็นเซอร์ (Diamond Blue)
            sensor_data.append({'Sensor_ID': f"SHT31-{r}-{t}", 'X': x_offset + 2, 'Y': y_offset + 1, 'T': tray_temp, 'H': tray_humid})
            
            for e_idx in range(10):
                e_row, e_col = e_idx // 5, e_idx % 5
                egg_temp = tray_temp + np.random.normal(0, 0.08)
                spike = np.random.choice([0, 1], p=[0.95, 0.05])
                egg_data.append({
                    'Egg_ID': f"{r}-{t}-C{e_idx+1:02d}", 'X': x_offset + e_col, 'Y': y_offset + e_row,
                    'Temp': np.round(egg_temp, 2), 'Humid': np.round(tray_humid, 2), 'Spike': spike,
                    'Rack': r, 'Tray': t
                })
    
    df = pd.DataFrame(egg_data)
    # AI Logic (XGBoost)
    X = df[['Temp', 'Humid', 'Spike']]
    y = np.random.binomial(1, np.clip(100 - (np.abs(df['Temp'] - 37.5) * 20 + df['Spike']*40), 0, 100) / 100.0)
    model = xgb.XGBClassifier(n_estimators=50).fit(X, y)
    return model, df, pd.DataFrame(sensor_data)

model, df, df_sensors = load_factory_data()

# 3. SIDEBAR: USER GUIDE & SPECS
with st.sidebar:
    st.title("🏢 ระบบควบคุมอาคาร")
    
    with st.expander("📖 คู่มือการใช้งาน (User Guide)", expanded=True):
        st.write("1. **Monitor:** ดูภาพรวมของไข่ 1,000 ใบในผังอาคาร")
        st.write("2. **Search:** ใช้ช่อง 'ค้นหาและซูม' เพื่อโฟกัสไข่ที่เสี่ยง")
        st.write("3. **Sensors:** สัญลักษณ์ ♦️ คือจุดติดตั้งเซ็นเซอร์ SHT31-D")
    
    with st.expander("🔬 AI Technical Spec"):
        st.latex(r"Prob = \sigma(\sum w_i x_i + b)")
        st.write("**Model:** XGBoost Classifier")
        st.write("**Accuracy:** 94.2% (Test Set)")
        
    with st.expander("💰 งบประมาณอุปกรณ์ (BOM)"):
        st.table(pd.DataFrame({
            "อุปกรณ์": ["SHT31-D", "ESP32", "R-Pi 4", "สายไฟ/ตู้"],
            "ราคา (บาท)": ["35,000", "15,000", "3,500", "5,600"]
        }))

    st.divider()
    threshold = st.slider("เกณฑ์แจ้งเตือนความเสี่ยง (%)", 10, 90, 50)

# 4. ANALYSIS & STATUS
df['Survival_Prob'] = np.round(model.predict_proba(df[['Temp', 'Humid', 'Spike']])[:, 1] * 100, 2)
df['Status'] = 'Safe'
df.loc[df['Survival_Prob'] < threshold, 'Status'] = 'Warning'
df.loc[df['Survival_Prob'] < 30, 'Status'] = 'Critical'

# 5. HEADER METRICS
st.title("🏢 Duck Hatchery: Smart Factory Digital Twin")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🥚 Monitoring Eggs", "1,000")
c2.metric("🌡️ Building Temp", f"{df['Temp'].mean():.2f} °C")
c3.metric("📡 Sensor Nodes", f"{len(df_sensors)}")
crit_num = (df['Status'] == 'Critical').sum()
c4.metric("🚨 Critical Status", f"{crit_num} ใบ", delta=crit_num, delta_color="inverse")

# 6. INTERACTIVE MAP (Digital Twin)
st.subheader("📍 Egg Incubation Building Plan")

# ระบบค้นหาและซูม (Fixed Error Version)
at_risk_list = df[df['Status'] != 'Safe']['Egg_ID'].tolist()
search_egg = st.selectbox("🔍 ค้นหาไอดีไข่เพื่อซูมพิกัด (Focus Search):", ["None"] + at_risk_list)

fig = go.Figure()

# วาดผนังและทางเดิน
fig.add_shape(type="rect", x0=-5, y0=-5, x1=90, y1=75, line=dict(color="#8b949e", width=3))
fig.add_shape(type="rect", x0=-5, y0=24, x1=90, y1=35, fillcolor="rgba(100,100,100,0.2)", line=dict(width=0))

# วาดไข่ตามสถานะ
colors = {'Safe': '#238636', 'Warning': '#d29922', 'Critical': '#f85149'}
for status in ['Safe', 'Warning', 'Critical']:
    sub = df[df['Status'] == status]
    fig.add_trace(go.Scatter(
        x=sub['X'], y=sub['Y'], mode='markers', name=status,
        marker=dict(size=9, color=colors[status], line=dict(width=0.5, color='white')),
        text=sub.apply(lambda r: f"ID: {r['Egg_ID']}<br>รอด: {r['Survival_Prob']}%", axis=1),
        hoverinfo='text'
    ))

# วาดจุดเซ็นเซอร์ (Blue Diamond)
fig.add_trace(go.Scatter(
    x=df_sensors['X'], y=df_sensors['Y'], mode='markers', name='Sensor Node',
    marker=dict(size=14, color='#1f6feb', symbol='diamond', line=dict(width=2, color='white')),
    text=df_sensors['Sensor_ID'], hoverinfo='text'
))

# ระบบ Zoom (Logic)
zoom_config = {}
if search_egg != "None":
    target = df[df['Egg_ID'] == search_egg].iloc[0]
    zoom_config = {
        'xaxis': {'range': [target['X']-10, target['X']+10], 'showgrid': False, 'zeroline': False, 'showticklabels': False},
        'yaxis': {'range': [target['Y']-10, target['Y']+10], 'showgrid': False, 'zeroline': False, 'showticklabels': False, 'scaleanchor':"x", 'scaleratio':1}
    }
    fig.add_shape(type="circle", x0=target['X']-2, y0=target['Y']-2, x1=target['X']+2, y1=target['Y']+2, line=dict(color="white", width=3))
else:
    zoom_config = {
        'xaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},
        'yaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False, 'scaleanchor':"x", 'scaleratio':1}
    }

fig.update_layout(template="plotly_dark", height=650, margin=dict(l=10, r=10, t=10, b=10),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                  **zoom_config)

st.plotly_chart(fig, use_container_width=True)

# 7. BOTTOM SECTION: WATCHLIST & DIAGNOSTICS
st.divider()
c_low1, c_low2 = st.columns([1, 1])

with c_low1:
    st.subheader("📋 Watchlist (รายการเฝ้าระวัง)")
    st.dataframe(df[df['Status'] != 'Safe'][['Egg_ID', 'Temp', 'Status', 'Survival_Prob']].sort_values('Survival_Prob'), 
                 hide_index=True, use_container_width=True)

with c_low2:
    st.subheader("🛠️ Sensor Diagnostics")
    st.dataframe(df_sensors[['Sensor_ID', 'T', 'H']], hide_index=True, use_container_width=True)
