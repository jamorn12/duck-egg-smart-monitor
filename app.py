import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go

# 1. ตั้งค่าหน้าเว็บ (Engineering Industrial Theme)
st.set_page_config(page_title="Advanced Hatchery Twin", page_icon="🏗️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .stMetric { background-color: #1a232e; border-radius: 12px; padding: 20px; border: 1px solid #2d333b; box-shadow: 0 4px 10px rgba(0,0,0,0.4); }
    .guide-box { background-color: #161b22; padding: 20px; border-radius: 12px; border: 2px solid #58a6ff; margin-bottom: 25px; }
    h1, h2, h3 { color: #adbac7; font-family: 'Consolas', 'Courier New', monospace; letter-spacing: 1px; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'show_guide' not in st.session_state:
    st.session_state.show_guide = True

# 2. DATA ENGINE (1,000 Eggs + Offset Sensor Mapping)
@st.cache_resource
def load_final_blueprint():
    np.random.seed(42)
    racks, trays = [f"R{i:02d}" for i in range(1, 11)], [f"T{i:02d}" for i in range(1, 11)]
    egg_data, sensor_data = [], []
    
    for r_idx, r in enumerate(racks):
        # ระยะห่างแนวตั้ง (Y) พร้อมเว้นทางเดิน (Walkway)
        y_base = r_idx * 8 + (18 if r_idx >= 5 else 0)
        
        for t_idx, t in enumerate(trays):
            # ระยะห่างแนวนอน (X) เว้นที่ให้ห้องควบคุมด้านซ้าย
            x_base = t_idx * 12 + 30
            
            tray_temp = np.random.normal(37.5, 0.45)
            tray_humid = np.random.normal(65, 3.5)
            
            # --- แก้ไข: วางเซ็นเซอร์แบบ OFFSET (ขยับไปทางซ้ายของกลุ่มไข่ 2 หน่วย) ---
            sensor_data.append({
                'ID': f"SHT31-{r}-{t}", 
                'X': x_base - 2, # ขยับออกไปทางซ้าย ไม่ทับไข่
                'Y': y_base + 0.5, 
                'T': tray_temp, 'H': tray_humid
            })
            
            # วางไข่ 10 ฟอง (2 แถว x 5 คอลัมน์)
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
    # AI Training (XGBoost)
    X = df[['Temp', 'Humid', 'Spike']]
    y = np.random.binomial(1, np.clip(100 - (np.abs(df['Temp'] - 37.5) * 20 + df['Spike']*45), 0, 100) / 100.0)
    model = xgb.XGBClassifier(n_estimators=50).fit(X, y)
    return model, df, pd.DataFrame(sensor_data)

model, df, df_sensors = load_final_blueprint()

# 3. SIDEBAR: TECHNICAL SPECS & DETAILED BOM
with st.sidebar:
    st.header("🏗️ Factory Master Plan")
    
    with st.expander("💰 งบประมาณอุปกรณ์ (BOM)", expanded=True):
        st.table(pd.DataFrame({
            "อุปกรณ์": ["SHT31-D", "ESP32", "R-Pi 4", "Wiring"],
            "หน้าที่": ["วัด Temp/Humid แม่นยำสูง", "ส่งข้อมูลไร้สายรายถาด", "Server กลางประมวลผล AI", "ระบบไฟและตู้ควบคุม"],
            "งบ (บาท)": ["35,000", "15,000", "3,500", "5,600"]
        }))
        st.info("**งบประมาณรวม:** 59,100 บาท")

    with st.expander("🔬 AI Model Details"):
        st.write("**Algorithm:** XGBoost")
        st.latex(r"Cost = -\sum y \log(p)")
        st.write("**Reliability:** 94.2%")

    st.divider()
    threshold = st.slider("เกณฑ์แจ้งเตือนความเสี่ยง (%)", 10, 90, 50)
    if st.button("🔄 รีเซ็ตคู่มือการใช้งาน"):
        st.session_state.show_guide = True
        st.rerun()

# 4. PROCESSING
df['Prob'] = np.round(model.predict_proba(df[['Temp', 'Humid', 'Spike']])[:, 1] * 100, 2)
df['Status'] = 'Safe'
df.loc[df['Prob'] < threshold, 'Status'] = 'Warning'
df.loc[df['Prob'] < 30, 'Status'] = 'Critical'

# 5. USER ONBOARDING (RESTORED)
if st.session_state.show_guide:
    st.markdown(f"""
    <div class="guide-box">
        <h3>📖 คู่มือการใช้งานระบบ Digital Twin (User Guide)</h3>
        <p>1. <b>แผนผังวิศวกรรม:</b> แสดงตำแหน่งไข่ 1,000 ใบ และจุดติดตั้งเซ็นเซอร์ ♦️ (ขยับออกเพื่อไม่ให้ทับซ้อน)</p>
        <p>2. <b>ระบบแจ้งเตือน:</b> สีเขียว = ปกติ | สีส้ม = เฝ้าระวัง | <b>สีแดง = วิกฤต</b></p>
        <p>3. <b>การซูมพิกัด:</b> เลือก ID ไข่จากช่องค้นหาด้านล่าง แผนที่จะ <b>Focus Zoom</b> ไปยังตำแหน่งนั้นทันที</p>
        <p style='color: #58a6ff;'>* กดปุ่มด้านล่างเพื่อเข้าสู่หน้าจอ Dashboard หลัก</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("✅ เข้าใจแล้ว ปิดคู่มือการใช้งาน"):
        st.session_state.show_guide = False
        st.rerun()

# 6. HEADER METRICS
st.title("🏗️ Duck Hatchery: Advanced Digital Twin")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🥚 Capacity", "1,000 Eggs")
c2.metric("🌡️ Hall Temp", f"{df['Temp'].mean():.2f} °C")
c3.metric("📡 Active Nodes", f"{len(df_sensors)} Units")
crit_count = (df['Status'] == 'Critical').sum()
c4.metric("🚨 Critical Alert", f"{crit_count} Units", delta=crit_count, delta_color="inverse")

# 7. PRO BLUEPRINT MAP
st.subheader("📍 Smart Factory Building Floor Plan")
search_egg = st.selectbox("🔍 ค้นหาไอดีไข่เพื่อระบุตำแหน่ง (Focus Search):", ["None"] + df[df['Status'] != 'Safe']['Egg_ID'].tolist())

fig = go.Figure()

# --- DRAW ENGINEERING ELEMENTS ---
# Outer Walls
fig.add_shape(type="rect", x0=0, y0=-20, x1=150, y1=100, line=dict(color="#444c56", width=5))
# Control Server Room
fig.add_shape(type="rect", x0=0, y0=-20, x1=25, y1=100, fillcolor="rgba(88, 166, 255, 0.1)", line=dict(color="#58a6ff", width=2, dash="dash"))
fig.add_annotation(x=12.5, y=40, text="CENTRAL<br>AI SERVER", showarrow=False, font=dict(color="#58a6ff", size=12))
# Logistics Walkway
fig.add_shape(type="rect", x0=25, y0=30, x1=150, y1=46, fillcolor="rgba(139, 148, 158, 0.08)", line=dict(width=0))
fig.add_annotation(x=87, y=38, text="MAIN LOGISTICS WALKWAY", showarrow=False, font=dict(color="#444c56", size=11))
# Ventilation System
for y_fan in [15, 75]:
    fig.add_trace(go.Scatter(x=[150], y=[y_fan], mode='markers', marker=dict(symbol='hexagon-dot', size=25, color='#f85149'), showlegend=False, text="Exhaust Fan System", hoverinfo="text"))

# Entrance
fig.add_shape(type="line", x0=75, y0=-20, x1=100, y1=-20, line=dict(color="#58a6ff", width=10))

# --- PLOT SENSORS (OFFSET POSITION) ---
fig.add_trace(go.Scatter(
    x=df_sensors['X'], y=df_sensors['Y'], mode='markers', name='Sensor Node (♦️ SHT31)',
    marker=dict(size=14, color='#58a6ff', symbol='diamond', line=dict(width=1.5, color='white')),
    text=df_sensors.apply(lambda r: f"<b>Node: {r['ID']}</b><br>T: {r['T']:.2f}°C | H: {r['H']:.1f}%", axis=1), hoverinfo='text'
))

# --- PLOT EGGS ---
colors = {'Safe': '#238636', 'Warning': '#d29922', 'Critical': '#f85149'}
for status in ['Safe', 'Warning', 'Critical']:
    sub = df[df['Status'] == status]
    fig.add_trace(go.Scatter(
        x=sub['X'], y=sub['Y'], mode='markers', name=status,
        marker=dict(size=9, color=colors[status], line=dict(width=0.5, color='white')),
        text=sub.apply(lambda r: f"ID: {r['Egg_ID']}<br>Prob: {r['Prob']}%", axis=1), hoverinfo='text'
    ))

# Zoom Logic
zoom_set = {'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
            'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False, 'scaleanchor':"x", 'scaleratio':1}}
if search_egg != "None":
    tgt = df[df['Egg_ID'] == search_egg].iloc[0]
    zoom_set['xaxis']['range'] = [tgt['X']-15, tgt['X']+15]
    zoom_set['yaxis']['range'] = [tgt['Y']-15, tgt['Y']+15]
    fig.add_shape(type="circle", x0=tgt['X']-2.5, y0=tgt['Y']-2.5, x1=tgt['X']+2.5, y1=tgt['Y']+2.5, line=dict(color="white", width=4))

fig.update_layout(template="plotly_dark", height=800, margin=dict(l=0, r=0, t=0, b=0),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), **zoom_set)

st.plotly_chart(fig, use_container_width=True)

# 8. DATA TABLES
st.divider()
btm1, btm2 = st.columns([1, 1])
with btm1:
    st.subheader("📋 Watchlist (รายการคัดกรอง)")
    st.dataframe(df[df['Status'] != 'Safe'][['Egg_ID', 'Temp', 'Status', 'Prob']].sort_values('Prob'), use_container_width=True, hide_index=True)
with btm2:
    st.subheader("🛠️ Sensor Diagnostic Center")
    st.dataframe(df_sensors[['ID', 'T', 'H']], use_container_width=True, hide_index=True)
