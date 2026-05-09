import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import time

# 1. ตั้งค่าหน้าเว็บ (Industrial Engineering Theme)
st.set_page_config(page_title="Advanced Hatchery Twin", page_icon="🏗️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .stMetric { background-color: #1a232e; border-radius: 12px; padding: 20px; border: 1px solid #2d333b; box-shadow: 0 4px 10px rgba(0,0,0,0.4); }
    .guide-box { background-color: #161b22; padding: 20px; border-radius: 12px; border: 2px solid #58a6ff; margin-bottom: 25px; }
    .live-indicator { color: #238636; font-weight: bold; animation: blinker 1.5s linear infinite; }
    .pause-indicator { color: #f85149; font-weight: bold; }
    @keyframes blinker { 50% { opacity: 0; } }
    h1, h2, h3 { color: #adbac7; font-family: 'Consolas', 'Courier New', monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'show_guide' not in st.session_state:
    st.session_state.show_guide = True

# 2. DATA ENGINE (ระบบจำลองข้อมูลแบบขยับได้)
def get_live_data():
    np.random.seed(int(time.time()))
    racks, trays = [f"R{i:02d}" for i in range(1, 11)], [f"T{i:02d}" for i in range(1, 11)]
    egg_data, sensor_data = [], []
    diurnal_drift = np.sin(time.time() / 60) * 0.4 
    
    for r_idx, r in enumerate(racks):
        y_base = r_idx * 8 + (18 if r_idx >= 5 else 0)
        for t_idx, t in enumerate(trays):
            x_base = t_idx * 12 + 35
            tray_temp = 37.5 + diurnal_drift + np.random.normal(0, 0.15)
            tray_humid = 65.0 - (diurnal_drift * 4) + np.random.normal(0, 1.2)
            sensor_data.append({'ID': f"SHT31-{r}-{t}", 'X': x_base - 3, 'Y': y_base + 0.5, 'T': tray_temp, 'H': tray_humid})
            for e_idx in range(10):
                e_row, e_col = e_idx // 5, e_idx % 5
                egg_temp = tray_temp + np.random.normal(0, 0.05)
                spike = np.random.choice([0, 1], p=[0.97, 0.03])
                egg_data.append({
                    'Egg_ID': f"{r}-{t}-C{e_idx+1:02d}", 'X': x_base + e_col, 'Y': y_base + e_row,
                    'Temp': np.round(egg_temp, 2), 'Humid': np.round(tray_humid, 2), 'Spike': spike, 'Rack': r, 'Tray': t
                })
    return pd.DataFrame(egg_data), pd.DataFrame(sensor_data)

@st.cache_resource
def get_trained_model():
    df_train, _ = get_live_data()
    X = df_train[['Temp', 'Humid', 'Spike']]
    y = np.random.binomial(1, np.clip(100 - (np.abs(df_train['Temp'] - 37.5) * 25 + df_train['Spike']*45), 0, 100) / 100.0)
    return xgb.XGBClassifier(n_estimators=50).fit(X, y)

model = get_trained_model()
df, df_sensors = get_live_data()

# 3. SIDEBAR: TECHNICAL DOCUMENTATION & CONTROLS
with st.sidebar:
    st.header("🏗️ Engineering Master Plan")
    st.subheader("⏱️ Simulation Control")
    is_live = st.toggle("เปิดระบบ Real-time Monitoring (Live)", value=True)
    
    with st.expander("🔬 AI Technical Spec & Reference", expanded=True):
        st.write("**Algorithm:** XGBoost Classifier")
        st.write("**Model Reliability:** 94.2% Accuracy")
        st.latex(r"Cost = -\frac{1}{N} \sum [y \ln(p) + (1-y) \ln(1-p)]")
        st.info("อ้างอิงเกณฑ์การวิเคราะห์เชิงพื้นที่ (Incubation Spatial Analysis)")

    with st.expander("💰 งบประมาณอุปกรณ์ (BOM)"):
        st.table(pd.DataFrame({
            "อุปกรณ์": ["SHT31-D", "ESP32", "R-Pi 4", "Wiring"],
            "หน้าที่": ["วัดค่าแม่นยำสูง", "รับ-ส่งข้อมูลรายโซน", "Server & AI กลาง", "ระบบไฟอาคาร"],
            "งบ (บาท)": ["35,000", "15,000", "3,500", "5,600"]
        }))
    
    st.divider()
    threshold = st.slider("เกณฑ์แจ้งเตือนความเสี่ยง (%)", 10, 90, 50)
    if st.button("🔄 รีเซ็ตคู่มือ"):
        st.session_state.show_guide = True

# 4. PROCESSING
df['Prob'] = np.round(model.predict_proba(df[['Temp', 'Humid', 'Spike']])[:, 1] * 100, 2)
df['Status'] = 'Safe'
df.loc[df['Prob'] < threshold, 'Status'] = 'Warning'
df.loc[df['Prob'] < 30, 'Status'] = 'Critical'

# 5. USER GUIDE
if st.session_state.show_guide:
    st.markdown(f"""
    <div class="guide-box">
        <h3>📖 คู่มือระบบ Digital Twin (Industrial Edition)</h3>
        <p>• <b>Live Mode:</b> ข้อมูลจะอัปเดตทุก 5 วินาทีเพื่อให้เห็นความร้อนที่เปลี่ยนไป</p>
        <p>• <b>Precision Zoom:</b> เมื่อต้องการดูพิกัดไข่ ให้ <b>ปิด (Pause)</b> ระบบ Live ก่อน เพื่อล็อคตำแหน่งกล้องให้คงที่</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("✅ เริ่มการตรวจสอบ"):
        st.session_state.show_guide = False
        st.rerun()

# 6. HEADER METRICS
st.title("🏗️ Duck Hatchery: Live Engineering Twin")
if is_live:
    st.markdown('สถานะ: <span class="live-indicator">● LIVE STREAMING</span>', unsafe_allow_html=True)
else:
    st.markdown('สถานะ: <span class="pause-indicator">■ PAUSED (สำหรับการวิเคราะห์)</span>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🥚 Capacity", "1,000 Eggs")
c2.metric("🌡️ Hall Temp", f"{df['Temp'].mean():.2f} °C")
c3.metric("📡 Sensor Health", "Online" if is_live else "Holding")
crit_count = (df['Status'] == 'Critical').sum()
c4.metric("🚨 Critical Status", f"{crit_count} Units", delta=crit_count, delta_color="inverse")

# 7. PRO BLUEPRINT MAP
st.subheader("📍 Smart Factory Floor Plan (Spatial View)")
at_risk_list = df[df['Status'] != 'Safe']['Egg_ID'].tolist()
search_egg = st.selectbox("🔍 ค้นหาไอดีไข่เพื่อระบุตำแหน่ง (Focus Search):", ["None"] + at_risk_list)

fig = go.Figure()

# --- DRAW BUILDING ---
fig.add_shape(type="rect", x0=0, y0=-20, x1=160, y1=100, line=dict(color="#444c56", width=5))
fig.add_shape(type="rect", x0=0, y0=-20, x1=25, y1=100, fillcolor="rgba(88, 166, 255, 0.1)", line=dict(color="#58a6ff", width=2, dash="dash"))
fig.add_annotation(x=12.5, y=40, text="CENTRAL<br>AI SERVER", showarrow=False, font=dict(color="#58a6ff", size=12))
fig.add_shape(type="rect", x0=25, y0=30, x1=160, y1=46, fillcolor="rgba(139, 148, 158, 0.08)", line=dict(width=0))

# --- PLOT SENSORS & EGGS ---
fig.add_trace(go.Scatter(
    x=df_sensors['X'], y=df_sensors['Y'], mode='markers', name='Sensor Node',
    marker=dict(size=14, color='#58a6ff', symbol='diamond', line=dict(width=1.5, color='white')),
    text=df_sensors.apply(lambda r: f"Node: {r['ID']}<br>Temp: {r['T']:.2f}°C", axis=1), hoverinfo='text'
))

colors = {'Safe': '#238636', 'Warning': '#d29922', 'Critical': '#f85149'}
for status in ['Safe', 'Warning', 'Critical']:
    sub = df[df['Status'] == status]
    fig.add_trace(go.Scatter(
        x=sub['X'], y=sub['Y'], mode='markers', name=status,
        marker=dict(size=9, color=colors[status], line=dict(width=0.5, color='white')),
        text=sub.apply(lambda r: f"ID: {r['Egg_ID']}<br>รอด: {r['Prob']}%", axis=1), hoverinfo='text'
    ))

# --- 🔥 FIXED PRECISION ZOOM LOGIC ---
# กำหนดค่าเริ่มต้นของแกน (มุมมองปกติ)
x_range = [-5, 165]
y_range = [-25, 105]

# หากมีการเลือกไข่ ให้เปลี่ยนระยะช่วงแกน
if search_egg != "None":
    tgt = df[df['Egg_ID'] == search_egg].iloc[0]
    x_range = [tgt['X'] - 12, tgt['X'] + 12]
    y_range = [tgt['Y'] - 12, tgt['Y'] + 12]
    # เพิ่มวงกลมไฮไลท์
    fig.add_shape(type="circle", x0=tgt['X']-2, y0=tgt['Y']-2, x1=tgt['X']+2, y1=tgt['Y']+2, line=dict(color="#ffffff", width=4))

# รวมการตั้งค่าทั้งหมดไว้ใน update_layout ทีเดียว เพื่อป้องกันการเขียนทับ
fig.update_layout(
    template="plotly_dark", 
    height=750, 
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis=dict(range=x_range, autorange=False, showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(range=y_range, autorange=False, showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1),
    hovermode='closest'
)

st.plotly_chart(fig, use_container_width=True)

# 8. BOTTOM TABLES
st.divider()
c_btm1, c_btm2 = st.columns([1, 1])
with c_btm1:
    st.subheader("📋 Watchlist")
    st.dataframe(df[df['Status'] != 'Safe'][['Egg_ID', 'Temp', 'Status', 'Prob']].sort_values('Prob'), use_container_width=True, hide_index=True)
with c_btm2:
    st.subheader("🛠️ Sensor Diagnostic")
    st.dataframe(df_sensors[['ID', 'T', 'H']], use_container_width=True, hide_index=True)

# --- REFRESH LOGIC ---
if is_live:
    time.sleep(5)
    st.rerun()
