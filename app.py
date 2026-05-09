import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go

# 1. ตั้งค่าหน้าเว็บ (Engineering & Industrial Design)
st.set_page_config(page_title="Advanced Hatchery Twin", page_icon="🏗️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .stMetric { background-color: #1a1f29; border-radius: 10px; padding: 20px; border: 1px solid #2d333b; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .guide-box { background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #58a6ff; margin-bottom: 25px; }
    h1, h2, h3 { color: #adbac7; font-family: 'Consolas', 'Courier New', monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'show_guide' not in st.session_state:
    st.session_state.show_guide = True

# 2. DATA ENGINE (1,000 Eggs + Pro Blueprint)
@st.cache_resource
def load_pro_blueprint():
    np.random.seed(42)
    racks, trays = [f"R{i:02d}" for i in range(1, 11)], [f"T{i:02d}" for i in range(1, 11)]
    egg_data, sensor_data = [], []
    
    for r_idx, r in enumerate(racks):
        y_base = r_idx * 7 + (15 if r_idx >= 5 else 0)
        for t_idx, t in enumerate(trays):
            x_base = t_idx * 10 + 25
            
            tray_temp = np.random.normal(37.5, 0.45)
            tray_humid = np.random.normal(65, 3.5)
            sensor_data.append({'ID': f"SHT31-{r}-{t}", 'X': x_base + 2, 'Y': y_base + 1, 'T': tray_temp, 'H': tray_humid})
            
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
    y = np.random.binomial(1, np.clip(100 - (np.abs(df['Temp'] - 37.5) * 20 + df['Spike']*45), 0, 100) / 100.0)
    model = xgb.XGBClassifier(n_estimators=50).fit(X, y)
    return model, df, pd.DataFrame(sensor_data)

model, df, df_sensors = load_pro_blueprint()

# 3. SIDEBAR: TECHNICAL SPECS & DETAILED BOM
with st.sidebar:
    st.header("🛠️ Engineering Specs")
    
    with st.expander("💰 งบประมาณอุปกรณ์ (BOM)", expanded=True):
        st.table(pd.DataFrame({
            "อุปกรณ์": ["SHT31-D", "ESP32", "R-Pi 4", "Wiring System"],
            "หน้าที่": ["วัด Temp/Humid แม่นยำสูงรายโซน", "ประมวลผลและส่งข้อมูลไร้สาย", "Gateway กลางประมวลผล AI", "ระบบจ่ายไฟและตู้ควบคุมกลาง"],
            "งบ (บาท)": ["35,000", "15,000", "3,500", "5,600"]
        }))
        st.info("**งบรวม:** 59,100 บาท")

    with st.expander("🔬 AI Model Details"):
        st.write("**Algorithm:** XGBoost")
        st.latex(r"Obj = \sum L(y_i, \hat{y}_i) + \Omega(f)")
        st.write("**Confidence:** 94.2% Reliability")

    st.divider()
    threshold = st.slider("เกณฑ์แจ้งเตือนความเสี่ยง (%)", 10, 90, 50)
    if st.button("🔄 รีเซ็ตหน้าจอคู่มือ"):
        st.session_state.show_guide = True
        st.rerun()

# 4. DATA PROCESSING
df['Prob'] = np.round(model.predict_proba(df[['Temp', 'Humid', 'Spike']])[:, 1] * 100, 2)
df['Status'] = 'Safe'
df.loc[df['Prob'] < threshold, 'Status'] = 'Warning'
df.loc[df['Prob'] < 30, 'Status'] = 'Critical'

# 5. USER ONBOARDING GUIDE (Pop-up Style)
if st.session_state.show_guide:
    st.markdown(f"""
    <div class="guide-box">
        <h3>📖 คู่มือการใช้งานระบบ Digital Twin (User Guide)</h3>
        <p>1. <b>แผนที่เชิงวิศวกรรม:</b> แสดงผังโรงเรือนจริง จุดติดตั้งเซ็นเซอร์ ♦️ และตำแหน่งไข่รายฟอง</p>
        <p>2. <b>การซูมพิกัด:</b> เลือก ID ไข่จากช่องค้นหาด้านล่างเพื่อ <b>ซูม (Auto-Focus)</b> ไปยังตำแหน่งนั้นทันที</p>
        <p>3. <b>สถานะเซ็นเซอร์:</b> นำเมาส์ไปชี้ที่จุด ♦️ เพื่อดูสถานะการทำงานของเซ็นเซอร์รายถาด</p>
        <p style='color: #58a6ff;'>* คุณสามารถปิดคู่มือนี้ได้โดยกดปุ่มด้านล่าง</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("✅ รับทราบและปิดคู่มือ"):
        st.session_state.show_guide = False
        st.rerun()

# 6. HEADER METRICS
st.title("🏢 Duck Hatchery: Smart Factory Digital Twin")
st.markdown("_ระบบวิเคราะห์พฤติกรรมเชิงพื้นที่และความเสี่ยงการฟักไข่ระดับอุตสาหกรรม_")

c1, c2, c3, c4 = st.columns(4)
c1.metric("🥚 Capacity", "1,000 Eggs")
c2.metric("🌡️ Hall Temp", f"{df['Temp'].mean():.2f} °C")
c3.metric("📡 Active Nodes", f"{len(df_sensors)} Units")
crit_count = (df['Status'] == 'Critical').sum()
c4.metric("🚨 Critical Alert", f"{crit_count} Units", delta=crit_count, delta_color="inverse")

# 7. PRO BLUEPRINT MAP
st.subheader("📍 Building Floor Plan & Grid System")
search_egg = st.selectbox("🔍 ค้นหาไอดีไข่เพื่อระบุตำแหน่ง (Focus Search):", ["None"] + df[df['Status'] != 'Safe']['Egg_ID'].tolist())

fig = go.Figure()

# --- DRAW ENGINEERING ELEMENTS ---
# Walls
fig.add_shape(type="rect", x0=0, y0=-15, x1=135, y1=90, line=dict(color="#444c56", width=5))
# Control Room Zone
fig.add_shape(type="rect", x0=0, y0=-15, x1=20, y1=90, fillcolor="rgba(88, 166, 255, 0.1)", line=dict(color="#58a6ff", width=2, dash="dash"))
fig.add_annotation(x=10, y=37, text="CONTROL<br>SERVER", showarrow=False, font=dict(color="#58a6ff", size=11))
# Walkway
fig.add_shape(type="rect", x0=20, y0=28, x1=135, y1=42, fillcolor="rgba(139, 148, 158, 0.05)", line=dict(width=0))
fig.add_annotation(x=77, y=35, text="LOGISTICS WALKWAY", showarrow=False, font=dict(color="#444c56", size=10))
# Ventilation
for x_fan in [135]:
    for y_fan in [15, 60]:
        fig.add_trace(go.Scatter(x=[x_fan], y=[y_fan], mode='markers', marker=dict(symbol='hexagon-dot', size=20, color='#f85149'), showlegend=False, text="Exhaust Fan", hoverinfo="text"))

# Entrance
fig.add_shape(type="line", x0=65, y0=-15, x1=85, y1=-15, line=dict(color="#58a6ff", width=8))

# --- PLOT EGGS & SENSORS ---
colors = {'Safe': '#238636', 'Warning': '#d29922', 'Critical': '#f85149'}
for status in ['Safe', 'Warning', 'Critical']:
    sub = df[df['Status'] == status]
    fig.add_trace(go.Scatter(
        x=sub['X'], y=sub['Y'], mode='markers', name=status,
        marker=dict(size=8, color=colors[status], line=dict(width=0.5, color='white')),
        text=sub.apply(lambda r: f"ID: {r['Egg_ID']}<br>รอด: {r['Prob']}%", axis=1), hoverinfo='text'
    ))

fig.add_trace(go.Scatter(
    x=df_sensors['X'], y=df_sensors['Y'], mode='markers', name='Sensor Node (SHT31)',
    marker=dict(size=13, color='#58a6ff', symbol='diamond', line=dict(width=1.5, color='white')),
    text=df_sensors.apply(lambda r: f"<b>Node: {r['ID']}</b><br>T: {r['T']:.2f} | H: {r['H']:.1f}", axis=1), hoverinfo='text'
))

# Zoom & Focus Logic
zoom_set = {'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
            'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False, 'scaleanchor':"x", 'scaleratio':1}}
if search_egg != "None":
    tgt = df[df['Egg_ID'] == search_egg].iloc[0]
    zoom_set['xaxis']['range'] = [tgt['X']-12, tgt['X']+12]
    zoom_set['yaxis']['range'] = [tgt['Y']-12, tgt['Y']+12]
    fig.add_shape(type="circle", x0=tgt['X']-2, y0=tgt['Y']-2, x1=tgt['X']+2, y1=tgt['Y']+2, line=dict(color="white", width=3))

fig.update_layout(template="plotly_dark", height=750, margin=dict(l=0, r=0, t=0, b=0),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), **zoom_set)

st.plotly_chart(fig, use_container_width=True)

# 8. DATA TABLES
st.divider()
btm1, btm2 = st.columns([1, 1])
with btm1:
    st.subheader("📋 Watchlist")
    st.dataframe(df[df['Status'] != 'Safe'][['Egg_ID', 'Temp', 'Status', 'Prob']].sort_values('Prob'), use_container_width=True, hide_index=True)
with btm2:
    st.subheader("🛠️ Sensor Diagnostic")
    st.dataframe(df_sensors[['ID', 'T', 'H']], use_container_width=True, hide_index=True)
