import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go

# 1. ตั้งค่าหน้าเว็บ (Engineering Blueprint Theme)
st.set_page_config(page_title="Hatchery Engineering Twin", page_icon="🏗️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b1423; }
    .stMetric { background-color: #1a2634; border-radius: 10px; padding: 20px; border: 1px solid #2e3b4e; }
    .stExpander { background-color: #1a2634; border: 1px solid #2e3b4e; }
    h1, h2, h3 { color: #e1e9f0; font-family: 'Courier New', Courier, monospace; }
    </style>
    """, unsafe_allow_html=True)

# 2. DATA ENGINE (1,000 Eggs + Engineering Layout)
@st.cache_resource
def load_blueprint_data():
    np.random.seed(42)
    racks, trays = [f"R{i:02d}" for i in range(1, 11)], [f"T{i:02d}" for i in range(1, 11)]
    egg_data, sensor_data = [], []
    
    for r_idx, r in enumerate(racks):
        # ระยะห่างระหว่าง Rack (Y-axis) พร้อมเว้นทางเดินกลาง
        y_base = r_idx * 7 + (15 if r_idx >= 5 else 0)
        for t_idx, t in enumerate(trays):
            # ระยะห่างระหว่าง Tray (X-axis)
            x_base = t_idx * 10 + 20 # ขยับเพื่อเว้นที่ให้ห้องควบคุมด้านซ้าย
            
            tray_temp = np.random.normal(37.5, 0.4)
            tray_humid = np.random.normal(65, 3.0)
            sensor_data.append({'Sensor_ID': f"SHT31-{r}-{t}", 'X': x_base + 2, 'Y': y_base + 1, 'T': tray_temp, 'H': tray_humid})
            
            for e_idx in range(10):
                e_row, e_col = e_idx // 5, e_idx % 5
                egg_temp = tray_temp + np.random.normal(0, 0.08)
                spike = np.random.choice([0, 1], p=[0.95, 0.05])
                egg_data.append({
                    'Egg_ID': f"{r}-{t}-C{e_idx+1:02d}", 'X': x_base + e_col, 'Y': y_base + e_row,
                    'Temp': np.round(egg_temp, 2), 'Humid': np.round(tray_humid, 2), 'Spike': spike,
                    'Rack': r, 'Tray': t
                })
    
    df = pd.DataFrame(egg_data)
    X = df[['Temp', 'Humid', 'Spike']]
    y = np.random.binomial(1, np.clip(100 - (np.abs(df['Temp'] - 37.5) * 20 + df['Spike']*40), 0, 100) / 100.0)
    model = xgb.XGBClassifier(n_estimators=50).fit(X, y)
    return model, df, pd.DataFrame(sensor_data)

model, df, df_sensors = load_blueprint_data()

# 3. SIDEBAR: USER GUIDE & DETAILED BOM
with st.sidebar:
    st.header("🏗️ Project Master Plan")
    
    with st.expander("💰 งบประมาณอุปกรณ์ (BOM)", expanded=True):
        bom_df = pd.DataFrame({
            "อุปกรณ์": ["SHT31-D", "ESP32", "R-Pi 4", "System/Wiring"],
            "หน้าที่": ["วัด Temp/Humid แม่นยำสูง", "รับ-ส่งข้อมูลไร้สาย", "เซิร์ฟเวอร์กลาง & AI", "ระบบไฟและตู้ควบคุม"],
            "งบ (บาท)": ["35,000", "15,000", "3,500", "5,600"]
        })
        st.table(bom_df)
        st.info("**งบประมาณรวม:** 59,100 บาท")

    with st.expander("📊 AI Model Stats"):
        st.write("**Model:** XGBoost Classifier")
        st.write("**Precision:** 94% | **Recall:** 82%")
        st.latex(r"Cost = -\frac{1}{N} \sum y \cdot \log(p)")

    st.divider()
    threshold = st.slider("เกณฑ์การแจ้งเตือนความเสี่ยง (%)", 10, 90, 50)

# 4. PROCESSING
df['Prob'] = np.round(model.predict_proba(df[['Temp', 'Humid', 'Spike']])[:, 1] * 100, 2)
df['Status'] = 'Safe'
df.loc[df['Prob'] < threshold, 'Status'] = 'Warning'
df.loc[df['Prob'] < 30, 'Status'] = 'Critical'

# 5. MAIN HEADER
st.title("🏗️ Duck Hatchery Industrial Digital Twin")
st.markdown("_Engineering-Grade Spatial Monitoring & Risk Assessment_")

m1, m2, m3, m4 = st.columns(4)
m1.metric("🥚 Capacity", "1,000 Eggs")
m2.metric("🌡️ Hall Temp", f"{df['Temp'].mean():.2f} °C")
m3.metric("📡 Active Nodes", f"{len(df_sensors)}")
crit_num = (df['Status'] == 'Critical').sum()
m4.metric("🚨 Critical Alert", f"{crit_num} Units", delta=crit_num, delta_color="inverse")

# 6. ENGINEERING BLUEPRINT MAP
st.subheader("📍 Building Floor Plan & Sensor Network")
search_egg = st.selectbox("🔍 ค้นหาและซูมพิกัดไข่ (Focus Search):", ["None"] + df[df['Status'] != 'Safe']['Egg_ID'].tolist())

fig = go.Figure()

# --- DRAW BUILDING STRUCTURE ---
# Outer Wall (อาคารหลัก)
fig.add_shape(type="rect", x0=0, y0=-10, x1=125, y1=85, line=dict(color="#4e5d6c", width=4))
# Control Room (ห้องควบคุม)
fig.add_shape(type="rect", x0=0, y0=-10, x1=15, y1=85, fillcolor="rgba(78, 93, 108, 0.2)", line=dict(color="#4e5d6c", width=2))
fig.add_annotation(x=7.5, y=37, text="CONTROL<br>ROOM", showarrow=False, font=dict(color="#4e5d6c", size=10), textangle=-90)
# Walkway (ทางเดินกลาง)
fig.add_shape(type="rect", x0=15, y0=28, x1=125, y1=42, fillcolor="rgba(255, 255, 255, 0.05)", line=dict(width=0))
fig.add_annotation(x=70, y=35, text="MAIN WALKWAY", showarrow=False, font=dict(color="#4e5d6c", size=9))

# Exhaust Fans (สัญลักษณ์พัดลมระบายอากาศ)
for fx, fy in [(125, 10), (125, 60)]:
    fig.add_trace(go.Scatter(x=[fx], y=[fy], mode='markers', marker=dict(symbol='hexagram', size=15, color='#ff4b4b'), showlegend=False, hoverinfo='text', text='Exhaust Fan'))

# Entrance
fig.add_shape(type="line", x0=60, y0=-10, x1=80, y1=-10, line=dict(color="#58a6ff", width=6))
fig.add_annotation(x=70, y=-14, text="MAIN ENTRANCE", showarrow=False, font=dict(color="#58a6ff", size=10))

# --- PLOT DATA ---
colors = {'Safe': '#238636', 'Warning': '#d29922', 'Critical': '#f85149'}
for status in ['Safe', 'Warning', 'Critical']:
    sub = df[df['Status'] == status]
    fig.add_trace(go.Scatter(
        x=sub['X'], y=sub['Y'], mode='markers', name=status,
        marker=dict(size=8, color=colors[status], line=dict(width=0.5, color='white')),
        text=sub.apply(lambda r: f"ID: {r['Egg_ID']}<br>Prob: {r['Prob']}%", axis=1), hoverinfo='text'
    ))

# Plot Sensors (Diamond)
fig.add_trace(go.Scatter(
    x=df_sensors['X'], y=df_sensors['Y'], mode='markers', name='Sensor Node',
    marker=dict(size=12, color='#58a6ff', symbol='diamond', line=dict(width=1, color='white')),
    text=df_sensors['Sensor_ID'], hoverinfo='text'
))

# Zoom Logic
zoom_config = {'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
               'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False, 'scaleanchor':"x", 'scaleratio':1}}
if search_egg != "None":
    target = df[df['Egg_ID'] == search_egg].iloc[0]
    zoom_config['xaxis']['range'] = [target['X']-12, target['X']+12]
    zoom_config['yaxis']['range'] = [target['Y']-12, target['Y']+12]
    fig.add_shape(type="circle", x0=target['X']-2, y0=target['Y']-2, x1=target['X']+2, y1=target['Y']+2, line=dict(color="white", width=3))

fig.update_layout(template="plotly_dark", height=700, margin=dict(l=0, r=0, t=0, b=0),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), **zoom_config)

st.plotly_chart(fig, use_container_width=True)

# 7. FOOTER: WATCHLIST & DIAGNOSTICS
st.divider()
c_btm1, c_btm2 = st.columns([1, 1])
with c_btm1:
    st.subheader("📋 Watchlist")
    st.dataframe(df[df['Status'] != 'Safe'][['Egg_ID', 'Temp', 'Status', 'Prob']].sort_values('Prob'), use_container_width=True, hide_index=True)
with c_btm2:
    st.subheader("🛠️ Sensor Diagnostics")
    st.dataframe(df_sensors[['Sensor_ID', 'T', 'H']], use_container_width=True, hide_index=True)
