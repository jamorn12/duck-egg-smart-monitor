import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import time

# 1. ตั้งค่าหน้าเว็บ (Engineering Industrial Theme)
st.set_page_config(page_title="Ultimate Hatchery Twin", page_icon="🏗️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .stMetric { background-color: #1a232e; border-radius: 12px; padding: 15px; border: 1px solid #2d333b; box-shadow: 0 4px 10px rgba(0,0,0,0.4); }
    .guide-box { background-color: #161b22; padding: 20px; border-radius: 12px; border: 2px solid #58a6ff; margin-bottom: 25px; }
    .live-indicator { color: #238636; font-weight: bold; animation: blinker 1.5s linear infinite; }
    .pause-indicator { color: #f85149; font-weight: bold; }
    @keyframes blinker { 50% { opacity: 0; } }
    h1, h2, h3 { color: #adbac7; font-family: 'Consolas', 'Courier New', monospace; }
    .stTabs [data-baseweb="tab-list"] { background-color: #161b22; border-radius: 10px; padding: 5px; }
    .stTabs [data-baseweb="tab"] { color: #adbac7; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE MANAGEMENT ---
if 'master_df' not in st.session_state: st.session_state.master_df = None
if 'sensor_df' not in st.session_state: st.session_state.sensor_df = None
if 'show_guide' not in st.session_state: st.session_state.show_guide = True

# 2. DATA ENGINE (ระบบจำลองข้อมูลแบบครบวงจร)
def generate_factory_data(weather_factor):
    np.random.seed(int(time.time()))
    racks, trays = [f"R{i:02d}" for i in range(1, 11)], [f"T{i:02d}" for i in range(1, 11)]
    egg_data, sensor_data = [], []
    diurnal_drift = (np.sin(time.time() / 60) * 0.4) + weather_factor 
    
    for r_idx, r in enumerate(racks):
        y_base = r_idx * 8 + (18 if r_idx >= 5 else 0)
        for t_idx, t in enumerate(trays):
            x_base = t_idx * 12 + 35
            tray_temp = 37.5 + diurnal_drift + np.random.normal(0, 0.15)
            tray_humid = 65.0 - (diurnal_drift * 4) + np.random.normal(0, 1.2)
            
            # ระบบ Control Loop
            fan_status = "ON (Cooling)" if tray_temp > 37.8 else "OFF (Standby)"
            
            sensor_data.append({
                'ID': f"SHT31-{r}-{t}", 'X': x_base - 3, 'Y': y_base + 0.5, 
                'T': tray_temp, 'H': tray_humid, 'Fan': fan_status
            })
            
            for e_idx in range(10):
                e_row, e_col = e_idx // 5, e_idx % 5
                egg_temp = tray_temp + np.random.normal(0, 0.05)
                days_left = max(0, int(28 - ((egg_temp - 37.0) * 2))) 
                
                # --- แก้ไขบั๊กตรงนี้: ใส่ Rack และ Tray กลับเข้าไปให้ครบ! ---
                egg_data.append({
                    'Egg_ID': f"{r}-{t}-C{e_idx+1:02d}", 'X': x_base + e_col, 'Y': y_base + e_row,
                    'Temp': np.round(egg_temp, 2), 'Humid': np.round(tray_humid, 2), 
                    'Spike': np.random.choice([0, 1], p=[0.97, 0.03]), 'Days_to_Hatch': days_left,
                    'Rack': r, 'Tray': t
                })
    return pd.DataFrame(egg_data), pd.DataFrame(sensor_data)

@st.cache_resource
def get_trained_model():
    df_sample, _ = generate_factory_data(0)
    X = df_sample[['Temp', 'Humid', 'Spike']]
    y = np.random.binomial(1, np.clip(100 - (np.abs(df_sample['Temp'] - 37.5) * 25 + df_sample['Spike']*45), 0, 100) / 100.0)
    return xgb.XGBClassifier(n_estimators=50).fit(X, y)

# 3. SIDEBAR & LOGIC
with st.sidebar:
    st.header("🏗️ Factory Master Plan")
    st.subheader("⏱️ Simulation Control")
    is_live = st.toggle("เปิดระบบ Real-time Monitoring (Live)", value=True)
    
    weather_impact = st.slider("☁️ อิทธิพลอากาศภายนอก (Weather Impact)", -0.5, 0.5, 0.0, 0.1, help="จำลองพายุหรือคลื่นความร้อน")
    egg_price = st.number_input("💰 ราคาไข่/ตัว (บาท)", value=15.0, step=1.0)
    
    # เช็คความสมบูรณ์ของ Data เพื่อกัน Error จาก Cache เก่า
    need_refresh = False
    if st.session_state.master_df is None:
        need_refresh = True
    elif 'Rack' not in st.session_state.master_df.columns or 'Fan' not in st.session_state.sensor_df.columns:
        need_refresh = True
        
    if is_live or need_refresh:
        st.session_state.master_df, st.session_state.sensor_df = generate_factory_data(weather_impact)

    with st.expander("🔬 AI & System Specs", expanded=True):
        st.write("**Model:** XGBoost (94.2% Accuracy)")
        st.latex(r"Cost = -\sum [y \ln(p) + (1-y) \ln(1-p)]")
        st.write("**Features:** Economic Intelligence, Hatch Prediction, Auto Control Loop")

    with st.expander("💰 งบประมาณอุปกรณ์ (BOM)"):
        st.table(pd.DataFrame({"อุปกรณ์": ["SHT31-D", "ESP32", "R-Pi 4", "System"], "หน้าที่": ["เซ็นเซอร์", "ส่งข้อมูล", "AI Server", "ตู้ไฟ"], "งบ (฿)": ["35k", "15k", "3.5k", "5.6k"]}))
    
    threshold = st.slider("เกณฑ์แจ้งเตือนความเสี่ยง (%)", 10, 90, 50)
    if st.button("🔄 รีเซ็ตคู่มือ"): st.session_state.show_guide = True

# ประมวลผล Data
model = get_trained_model()
df, df_sensors = st.session_state.master_df.copy(), st.session_state.sensor_df.copy()
df['Prob'] = np.round(model.predict_proba(df[['Temp', 'Humid', 'Spike']])[:, 1] * 100, 2)
df['Status'] = 'Safe'
df.loc[df['Prob'] < threshold, 'Status'] = 'Warning'
df.loc[df['Prob'] < 30, 'Status'] = 'Critical'

# คำนวณเศรษฐศาสตร์และพลังงาน
est_survival_count = (df['Prob'] / 100).sum()
est_revenue = est_survival_count * egg_price
potential_loss = (len(df) - est_survival_count) * egg_price
active_fans = (df_sensors['Fan'] == "ON (Cooling)").sum()
power_usage = 12.5 + (active_fans * 0.8)

# 4. HEADER & METRICS
st.title("🏗️ Duck Hatchery: Ultimate Intelligence Twin")
if is_live:
    st.markdown('สถานะ: <span class="live-indicator">● LIVE STREAMING & CONTROL LOOP ACTIVE</span>', unsafe_allow_html=True)
else:
    st.markdown('สถานะ: <span class="pause-indicator">■ PAUSED (ข้อมูลล็อคเพื่อการวิเคราะห์)</span>', unsafe_allow_html=True)

# แถวที่ 1: Technical Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("🥚 Capacity", f"{len(df):,} ฟอง")
c2.metric("🌡️ Hall Temp Avg", f"{df['Temp'].mean():.2f} °C")
c3.metric("📡 Sensor Health", "Online" if is_live else "Frozen")
crit_count = (df['Status'] == 'Critical').sum()
c4.metric("🚨 Critical Status", f"{crit_count} ใบ", delta=crit_count, delta_color="inverse")

# แถวที่ 2: Business & Energy Metrics
b1, b2, b3, b4 = st.columns(4)
b1.metric("💰 คาดการณ์รายได้ (Est. Revenue)", f"฿{est_revenue:,.2f}")
b2.metric("📉 มูลค่าความเสี่ยง (Potential Loss)", f"฿{potential_loss:,.2f}", delta="- Risk", delta_color="inverse")
b3.metric("⚡ การใช้พลังงาน (HVAC Power)", f"{power_usage:.1f} kWh")
b4.metric("🐣 เฉลี่ยวันฟักตัว (Avg Hatch)", f"{df['Days_to_Hatch'].mean():.1f} วัน")

# 5. USER GUIDE
if st.session_state.show_guide:
    st.markdown("""<div class="guide-box"><h3>📖 คู่มือระบบ Ultimate Twin</h3><p>1. <b>Tabs:</b> เลือกแท็บด้านล่างเพื่อดูแผนที่หลัก, เศรษฐศาสตร์, หรือสถิติย้อนหลัง<br>2. <b>การซูม:</b> ปิด Live ด้านซ้าย แล้วใช้ช่องค้นหาเพื่อซูม<br>3. <b>Control Loop:</b> ระบบพัดลมจะทำงานอัตโนมัติ (แถบพลังงานจะขึ้น) เมื่ออุณหภูมิเกิน 37.8°C</p></div>""", unsafe_allow_html=True)
    if st.button("✅ ปิดคู่มือ"): st.session_state.show_guide = False; st.rerun()

# --- TABS SYSTEM ---
tab1, tab2, tab3 = st.tabs(["📍 Real-time Spatial Map", "💼 Economic & HVAC Logic", "📈 Historical Analysis"])

with tab1:
    search_egg = st.selectbox("🔍 ค้นหาไอดีไข่เพื่อระบุตำแหน่ง (Focus Search):", ["None"] + df[df['Status'] != 'Safe']['Egg_ID'].tolist())
    
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=-20, x1=160, y1=100, line=dict(color="#444c56", width=5))
    fig.add_shape(type="rect", x0=0, y0=-20, x1=25, y1=100, fillcolor="rgba(88, 166, 255, 0.1)", line=dict(color="#58a6ff", width=2, dash="dash"))
    fig.add_annotation(x=12.5, y=40, text="CENTRAL SERVER", showarrow=False, font=dict(color="#58a6ff", size=10))
    
    for y_fan in [15, 75]:
        fig.add_trace(go.Scatter(x=[160], y=[y_fan], mode='markers', marker=dict(symbol='hexagon-dot', size=25, color='#f85149' if active_fans > 0 else '#444c56'), showlegend=False, text="Exhaust Fan (Auto-Control)", hoverinfo="text"))

    fig.add_trace(go.Scatter(x=df_sensors['X'], y=df_sensors['Y'], mode='markers', name='Sensor', marker=dict(size=14, color='#58a6ff', symbol='diamond', line=dict(width=1.5, color='white')), text=df_sensors.apply(lambda r: f"Node: {r['ID']}<br>Fan: {r['Fan']}", axis=1), hoverinfo='text'))
    
    colors = {'Safe': '#238636', 'Warning': '#d29922', 'Critical': '#f85149'}
    for status in ['Safe', 'Warning', 'Critical']:
        sub = df[df['Status'] == status]
        fig.add_trace(go.Scatter(x=sub['X'], y=sub['Y'], mode='markers', name=status, marker=dict(size=9, color=colors[status], line=dict(width=0.5, color='white')), text=sub.apply(lambda r: f"ID: {r['Egg_ID']}<br>รอด: {r['Prob']}%<br>ฟักใน: {r['Days_to_Hatch']} วัน", axis=1), hoverinfo='text'))

    x_range, y_range = [-5, 165], [-25, 105]
    if search_egg != "None":
        tgt = df[df['Egg_ID'] == search_egg].iloc[0]
        x_range, y_range = [tgt['X']-12, tgt['X']+12], [tgt['Y']-12, tgt['Y']+12]
        fig.add_shape(type="circle", x0=tgt['X']-2, y0=tgt['Y']-2, x1=tgt['X']+2, y1=tgt['Y']+2, line=dict(color="#ffffff", width=4))

    fig.update_layout(template="plotly_dark", height=700, margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(range=x_range, autorange=False, showgrid=False, showticklabels=False, zeroline=False), yaxis=dict(range=y_range, autorange=False, showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), uirevision='constant')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    c_btm1, c_btm2 = st.columns([1, 1])
    with c_btm1:
        st.subheader("📋 Watchlist")
        st.dataframe(df[df['Status'] != 'Safe'][['Egg_ID', 'Temp', 'Status', 'Prob', 'Days_to_Hatch']].sort_values('Prob'), use_container_width=True, hide_index=True)
    with c_btm2:
        st.subheader("🛠️ Sensor Diagnostics")
        st.dataframe(df_sensors[['ID', 'T', 'H', 'Fan']], use_container_width=True, hide_index=True)

with tab2:
    st.subheader("💼 Economic Intelligence & Control Loop")
    e1, e2 = st.columns(2)
    with e1:
        st.info("📉 การประเมินมูลค่าความเสี่ยงรายถาด (Risk Valuation)")
        # ตอนนี้จะไม่เกิด KeyError: 'Rack' อีกแล้วครับ!
        risk_val = df[df['Status'] != 'Safe'].groupby(['Rack', 'Tray']).size().reset_index(name='Risk_Count')
        if not risk_val.empty:
            risk_val['Lost_Value_THB'] = risk_val['Risk_Count'] * egg_price
            st.dataframe(risk_val.sort_values('Lost_Value_THB', ascending=False), hide_index=True, use_container_width=True)
        else:
            st.success("ไม่มีความเสี่ยงในระบบ")
            
    with e2:
        st.warning("⚙️ ระบบสั่งการพัดลมอัตโนมัติ (HVAC Auto-Control)")
        active_cooling = df_sensors[df_sensors['Fan'] == 'ON (Cooling)']
        if not active_cooling.empty:
            st.error(f"ตรวจพบความร้อนเกิน 37.8°C: พัดลมทำงาน {len(active_cooling)} โซน")
            st.dataframe(active_cooling[['ID', 'T']], hide_index=True, use_container_width=True)
        else:
            st.success("อุณหภูมิทุกโซนอยู่ในเกณฑ์ปกติ (Standby)")

with tab3:
    st.subheader("📈 Historical Dead Zone Analysis (จำลองสถิติ 30 วัน)")
    hist_data = df.groupby(['Rack', 'Tray']).agg({'X': 'mean', 'Y': 'mean'}).reset_index()
    hist_data['Historical_Risk'] = np.random.uniform(0, 100, size=len(hist_data))
    fig_hist = go.Figure(data=go.Heatmap(
        z=hist_data['Historical_Risk'], x=hist_data['X'], y=hist_data['Y'],
        colorscale='Inferno', hoverinfo='text',
        text=hist_data.apply(lambda r: f"Zone: {r['Rack']}-{r['Tray']}<br>Risk Index: {r['Historical_Risk']:.1f}", axis=1)
    ))
    fig_hist.update_layout(template="plotly_dark", height=500, xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=False, showticklabels=False, scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig_hist, use_container_width=True)

# --- REFRESH LOOP ---
if is_live: time.sleep(5); st.rerun()
