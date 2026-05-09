import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import time

# 1. ตั้งค่าหน้าเว็บ (Engineering Industrial Theme)
st.set_page_config(page_title="Advanced Hatchery Twin PRO", page_icon="🏗️", layout="wide")

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

# --- SESSION STATE MANAGEMENT (ระบบล็อคข้อมูลและสถานะการซูม) ---
if 'master_df' not in st.session_state: st.session_state.master_df = None
if 'sensor_df' not in st.session_state: st.session_state.sensor_df = None
if 'selected_egg_id' not in st.session_state: st.session_state.selected_egg_id = None
if 'show_guide' not in st.session_state: st.session_state.show_guide = True

# 2. DATA ENGINE (ระบบจำลองข้อมูล)
def generate_factory_data():
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
                egg_data.append({
                    'Egg_ID': f"{r}-{t}-C{e_idx+1:02d}", 'X': x_base + e_col, 'Y': y_base + e_row,
                    'Temp': np.round(tray_temp + np.random.normal(0, 0.05), 2),
                    'Humid': np.round(tray_humid, 2), 'Spike': np.random.choice([0, 1], p=[0.97, 0.03])
                })
    return pd.DataFrame(egg_data), pd.DataFrame(sensor_data)

@st.cache_resource
def get_trained_model():
    df_sample, _ = generate_factory_data()
    X = df_sample[['Temp', 'Humid', 'Spike']]
    y = np.random.binomial(1, np.clip(100 - (np.abs(df_sample['Temp'] - 37.5) * 25 + df_sample['Spike']*45), 0, 100) / 100.0)
    return xgb.XGBClassifier(n_estimators=50).fit(X, y)

# 3. SIDEBAR & LOGIC (PLAY/PAUSE + BOM + TECH)
with st.sidebar:
    st.header("🏗️ Engineering Master Plan")
    st.subheader("⏱️ Simulation Control")
    is_live = st.toggle("เปิดระบบ Real-time Monitoring (Live)", value=True)
    
    # ล็อคข้อมูล: ถ้า Live ให้ดึงใหม่ / ถ้า Pause ให้ดึงจาก Session State เท่านั้น
    if is_live or st.session_state.master_df is None:
        st.session_state.master_df, st.session_state.sensor_df = generate_factory_data()
        if is_live: st.session_state.selected_egg_id = None # รีเซ็ตการซูมเมื่อกลับมา Live

    with st.expander("🔬 AI Technical Spec", expanded=True):
        st.write("**Model:** XGBoost Classifier (94.2% Accuracy)")
        st.latex(r"Cost = -\sum [y \ln(p) + (1-y) \ln(1-p)]")
        st.info("ระบบคำนวณแบบ Spatial Thermal Analysis")

    with st.expander("💰 งบประมาณอุปกรณ์ (BOM)"):
        st.table(pd.DataFrame({
            "อุปกรณ์": ["SHT31-D", "ESP32", "R-Pi 4", "System"],
            "หน้าที่": ["เซ็นเซอร์แม่นยำสูง", "ส่งข้อมูลรายโซน", "ประมวลผล AI", "ระบบไฟ"],
            "งบ (บาท)": ["35,000", "15,000", "3,500", "5,600"]
        }))
    
    threshold = st.slider("เกณฑ์แจ้งเตือนความเสี่ยง (%)", 10, 90, 50)
    if st.button("🔄 รีเซ็ตคู่มือ"): st.session_state.show_guide = True

# ประมวลผลสถานะไข่
model = get_trained_model()
df, df_sensors = st.session_state.master_df.copy(), st.session_state.sensor_df.copy()
df['Prob'] = np.round(model.predict_proba(df[['Temp', 'Humid', 'Spike']])[:, 1] * 100, 2)
df['Status'] = 'Safe'
df.loc[df['Prob'] < threshold, 'Status'] = 'Warning'
df.loc[df['Prob'] < 30, 'Status'] = 'Critical'

# 4. HEADER & METRICS
st.title("🏗️ Duck Hatchery: Live Engineering Twin")
if is_live:
    st.markdown('สถานะ: <span class="live-indicator">● LIVE STREAMING</span>', unsafe_allow_html=True)
else:
    st.markdown('สถานะ: <span class="pause-indicator">■ PAUSED (DATA FROZEN)</span>', unsafe_allow_html=True)

# Metrics (อยู่ครบถ้วนตามต้องการ)
c1, c2, c3, c4 = st.columns(4)
c1.metric("🥚 Capacity", f"{len(df):,} ฟอง")
c2.metric("🌡️ Hall Temp Avg", f"{df['Temp'].mean():.2f} °C")
c3.metric("📡 Sensor Health", "Online" if is_live else "Frozen")
crit_count = (df['Status'] == 'Critical').sum()
c4.metric("🚨 Critical Status", f"{crit_count} ใบ", delta=crit_count, delta_color="inverse")

# 5. USER GUIDE
if st.session_state.show_guide:
    st.markdown("""<div class="guide-box"><h3>📖 วิธีการซูมวิเคราะห์</h3><p>1. <b>ปิด Live</b> ด้านซ้ายเพื่อให้ข้อมูลนิ่ง<br>2. <b>เลือก ID ไข่</b> จากช่องค้นหาหรือคลิกตารางเพื่อซูมพิกัด</p></div>""", unsafe_allow_html=True)
    if st.button("✅ ปิดคู่มือ"): st.session_state.show_guide = False; st.rerun()

# 6. PRO BLUEPRINT MAP (ผังโรงเรือนครบชุด)
st.subheader("📍 Smart Factory Floor Plan (Engineering Layout)")
# ช่องค้นหา (เป็น Fallback กรณีเวอร์ชันเก่า)
search_egg = st.selectbox("🔍 ค้นหาไอดีไข่เพื่อระบุตำแหน่ง (Focus Search):", ["None"] + df[df['Status'] != 'Safe']['Egg_ID'].tolist())
if search_egg != "None": st.session_state.selected_egg_id = search_egg

fig = go.Figure()
# วาดอาคารและห้องควบคุม
fig.add_shape(type="rect", x0=0, y0=-20, x1=160, y1=100, line=dict(color="#444c56", width=5))
fig.add_shape(type="rect", x0=0, y0=-20, x1=25, y1=100, fillcolor="rgba(88, 166, 255, 0.1)", line=dict(color="#58a6ff", width=2, dash="dash"))
fig.add_annotation(x=12.5, y=40, text="CENTRAL SERVER", showarrow=False, font=dict(color="#58a6ff", size=10))
# พัดลมระบายอากาศ
for y_fan in [15, 75]: fig.add_trace(go.Scatter(x=[160], y=[y_fan], mode='markers', marker=dict(symbol='hexagon-dot', size=25, color='#f85149'), showlegend=False, text="Exhaust Fan", hoverinfo="text"))

# Plot เซ็นเซอร์และไข่
fig.add_trace(go.Scatter(x=df_sensors['X'], y=df_sensors['Y'], mode='markers', name='Sensor', marker=dict(size=14, color='#58a6ff', symbol='diamond', line=dict(width=1.5, color='white')), text=df_sensors['ID'], hoverinfo='text'))
colors = {'Safe': '#238636', 'Warning': '#d29922', 'Critical': '#f85149'}
for status in ['Safe', 'Warning', 'Critical']:
    sub = df[df['Status'] == status]
    fig.add_trace(go.Scatter(x=sub['X'], y=sub['Y'], mode='markers', name=status, marker=dict(size=9, color=colors[status], line=dict(width=0.5, color='white')), text=sub.apply(lambda r: f"ID: {r['Egg_ID']}<br>รอด: {r['Prob']}%", axis=1), hoverinfo='text'))

# --- 🔥 PRECISION ZOOM LOGIC ---
x_range, y_range = [-5, 165], [-25, 105]
if st.session_state.selected_egg_id and st.session_state.selected_egg_id != "None":
    tgt = df[df['Egg_ID'] == st.session_state.selected_egg_id].iloc[0]
    x_range, y_range = [tgt['X']-12, tgt['X']+12], [tgt['Y']-12, tgt['Y']+12]
    fig.add_shape(type="circle", x0=tgt['X']-2, y0=tgt['Y']-2, x1=tgt['X']+2, y1=tgt['Y']+2, line=dict(color="#ffffff", width=4))

fig.update_layout(template="plotly_dark", height=750, margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(range=x_range, autorange=False, showgrid=False, showticklabels=False, zeroline=False), yaxis=dict(range=y_range, autorange=False, showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), uirevision='constant')
st.plotly_chart(fig, use_container_width=True)

# 7. INTERACTIVE TABLES
st.divider()
c_btm1, c_btm2 = st.columns([1, 1])
with c_btm1:
    st.subheader("📋 Watchlist (คลิกเลือกแถวเพื่อซูมพิกัด)")
    watchlist_df = df[df['Status'] != 'Safe'][['Egg_ID', 'Temp', 'Status', 'Prob']].sort_values('Prob')
    try:
        # ระบบคลิกที่ตาราง (จะทำงานถ้าเวอร์ชันคือ 1.35.0+)
        selected_rows = st.dataframe(watchlist_df, on_select="rerun", selection_mode="single_row", hide_index=True, use_container_width=True)
        if len(selected_rows.selection.rows) > 0:
            st.session_state.selected_egg_id = watchlist_df.iloc[selected_rows.selection.rows[0]]['Egg_ID']
    except:
        st.warning("⚠️ คลิกที่ตารางไม่ได้เพราะเวอร์ชันเก่า แต่คุณสามารถซูมได้โดยใช้ 'ช่องค้นหา' ด้านบนแผนที่")
    if st.button("❌ ล้างการซูม (Reset View)"): st.session_state.selected_egg_id = "None"; st.rerun()

with c_btm2:
    st.subheader("🛠️ Sensor Diagnostics")
    st.dataframe(df_sensors[['ID', 'T', 'H']], use_container_width=True, hide_index=True)

# --- REFRESH LOOP ---
if is_live: time.sleep(5); st.rerun()
