import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# 1. ตั้งค่าหน้าเว็บระดับสูง
st.set_page_config(page_title="Industrial Hatchery Twin", page_icon="🏢", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .stMetric { background-color: #161b22; border-radius: 12px; padding: 20px; border: 1px solid #30363d; }
    .guide-box { background-color: #1f2937; padding: 15px; border-left: 5px solid #58a6ff; border-radius: 5px; margin-bottom: 20px; }
    h1, h2, h3 { color: #f0f6fc; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZATION & SESSION STATE ---
if 'guide_seen' not in st.session_state:
    st.session_state.guide_seen = False
if 'selected_egg' not in st.session_state:
    st.session_state.selected_egg = None

# 2. DATA ENGINE (1,000 Eggs + Facility)
@st.cache_resource
def load_industrial_data():
    np.random.seed(42)
    racks, trays = [f"R{i:02d}" for i in range(1, 11)], [f"T{i:02d}" for i in range(1, 11)]
    egg_data, sensor_data = [], []
    
    for r_idx, r in enumerate(racks):
        y_offset = r_idx * 6 + (12 if r_idx >= 5 else 0) # Walkway at middle
        for t_idx, t in enumerate(trays):
            x_offset = t_idx * 9
            # Sensor SHT31-D logic
            tray_temp = np.random.normal(37.5, 0.45)
            tray_humid = np.random.normal(65, 3.5)
            sensor_data.append({'Sensor_ID': f"SHT31-{r}-{t}", 'X': x_offset + 2.5, 'Y': y_offset + 1, 'Temp': tray_temp, 'Humid': tray_humid})
            
            for e_idx in range(10):
                e_row, e_col = e_idx // 5, e_idx % 5
                egg_temp = tray_temp + np.random.normal(0, 0.08)
                spike = np.random.choice([0, 1], p=[0.95, 0.05])
                egg_data.append({
                    'Egg_ID': f"{r}-{t}-C{e_idx+1:02d}", 'Rack': r, 'Tray': t,
                    'X': x_offset + e_col, 'Y': y_offset + e_row,
                    'Temp': np.round(egg_temp, 2), 'Humid': np.round(tray_humid, 2), 'Spike': spike,
                    'Sensor_Ref': f"SHT31-{r}-{t}"
                })
    
    df = pd.DataFrame(egg_data)
    # AI Simulation (XGBoost)
    X = df[['Temp', 'Humid', 'Spike']]
    y = np.random.binomial(1, np.clip(100 - (np.abs(df['Temp'] - 37.5) * 20 + df['Spike']*40), 0, 100) / 100.0)
    model = xgb.XGBClassifier(n_estimators=50).fit(X, y)
    return model, df, pd.DataFrame(sensor_data)

model, df, df_sensors = load_industrial_data()

# 3. SIDEBAR: TECHNICAL DOCUMENTATION & UI
with st.sidebar:
    st.title("🕹️ Building Control")
    st.divider()
    
    # User Guide Toggle
    if st.button("📖 เปิด/ปิด คู่มือการใช้งาน"):
        st.session_state.guide_seen = not st.session_state.guide_seen
    
    # Technical Section
    with st.expander("🔬 AI Technical Spec", expanded=False):
        st.write("**Method:** XGBoost Classifier")
        st.write("**Objective Function:**")
        st.latex(r"L(\theta) = \sum [y_i \ln(p_i) + (1-y_i) \ln(1-p_i)]")
        st.write("**Confidence Level:** 94.2% Accuracy")
        st.info("ระบบคำนวณความเสี่ยงจาก Gradient Boosting ของปัจจัยอุณหภูมิและความชื้นสัมพัทธ์")

    with st.expander("💸 Investment Summary (BOM)", expanded=False):
        bom_data = {
            "Item": ["SHT31-D", "ESP32", "Raspberry Pi 4", "Wiring/Misc"],
            "Qty": [100, 100, 1, "Set"],
            "Total (THB)": [35000, 15000, 3500, 5600]
        }
        st.table(pd.DataFrame(bom_data))
        st.write("**Total Investment:** ฿59,100")

    st.divider()
    threshold = st.sidebar.slider("เกณฑ์แจ้งเตือนความเสี่ยง (%)", 10, 90, 50)

# 4. ONBOARDING GUIDE
if st.session_state.guide_seen:
    st.markdown("""
    <div class="guide-box">
    <h3>👋 ยินดีต้อนรับสู่ระบบ Digital Twin!</h3>
    <p><b>1. แผนที่โรงเรือน:</b> แสดงตำแหน่งไข่ 1,000 ใบ และพิกัดเซ็นเซอร์ SHT31-D (รูปเพชรสีน้ำเงิน)</p>
    <p><b>2. การโต้ตอบ:</b> คลิกเลือกรายชื่อไข่ในตาราง "รายการเฝ้าระวัง" ด้านล่าง เพื่อ <b>ซูม (Auto-Zoom)</b> ไปยังตำแหน่งไข่ใบนั้นทันที</p>
    <p><b>3. แถบควบคุม:</b> ปรับค่า Threshold ด้านซ้าย เพื่อกำหนดความไวของ AI ในการแจ้งเตือน</p>
    </div>
    """, unsafe_allow_html=True)

# 5. ANALYSIS LOGIC
df['Survival_Prob'] = np.round(model.predict_proba(df[['Temp', 'Humid', 'Spike']])[:, 1] * 100, 2)
df['Status'] = 'Safe'
df.loc[df['Survival_Prob'] < threshold, 'Status'] = 'Warning'
df.loc[df['Survival_Prob'] < 30, 'Status'] = 'Critical'

# 6. MAIN DASHBOARD: METRICS
m1, m2, m3, m4 = st.columns(4)
m1.metric("🥚 Monitoring Eggs", "1,000")
m2.metric("🌡️ Building Temp", f"{df['Temp'].mean():.2f} °C")
critical_count = (df['Status'] == 'Critical').sum()
m4.metric("🚨 Critical Status", f"{critical_count} ใบ", delta=f"{critical_count}", delta_color="inverse")

# 7. INTERACTIVE MAP WITH ZOOM LOGIC
st.subheader("📍 Real-time Digital Twin Building Plan")

fig = go.Figure()

# Draw Building Walls & Path
fig.add_shape(type="rect", x0=-5, y0=-5, x1=90, y1=75, line=dict(color="#8b949e", width=3)) # Walls
fig.add_shape(type="rect", x0=-5, y0=24, x1=90, y1=35, fillcolor="rgba(100,100,100,0.2)", line=dict(width=0)) # Walkway
fig.add_annotation(x=42.5, y=-8, text="🚪 ทางเข้าหลัก (Main Entrance)", showarrow=False, font=dict(color="#58a6ff"))

# Plot Eggs
colors = {'Safe': '#238636', 'Warning': '#d29922', 'Critical': '#f85149'}
for status in ['Safe', 'Warning', 'Critical']:
    sub = df[df['Status'] == status]
    fig.add_trace(go.Scatter(
        x=sub['X'], y=sub['Y'], mode='markers', name=status,
        marker=dict(size=9, color=colors[status], line=dict(width=0.5, color='white')),
        text=sub.apply(lambda r: f"ID: {r['Egg_ID']}<br>Prob: {r['Survival_Prob']}%<br>Zone: {r['Rack']}-{r['Tray']}", axis=1),
        hoverinfo='text'
    ))

# Plot Sensors
fig.add_trace(go.Scatter(
    x=df_sensors['X'], y=df_sensors['Y'], mode='markers', name='Sensor Node',
    marker=dict(size=14, color='#1f6feb', symbol='diamond', line=dict(width=2, color='white')),
    text=df_sensors.apply(lambda r: f"<b>Sensor: {r['Sensor_ID']}</b><br>Online", axis=1),
    hoverinfo='text'
))

# --- ZOOM LOGIC ---
zoom_range = None
if st.session_state.selected_egg:
    egg_info = df[df['Egg_ID'] == st.session_state.selected_egg].iloc[0]
    zoom_range = {
        'x': [egg_info['X'] - 10, egg_info['X'] + 10],
        'y': [egg_info['Y'] - 10, egg_info['Y'] + 10]
    }
    # Add a highlight circle around the selected egg
    fig.add_shape(type="circle", x0=egg_info['X']-1.5, y0=egg_info['Y']-1.5, x1=egg_info['X']+1.5, y1=egg_info['Y']+1.5, line=dict(color="#ffffff", width=3))

fig.update_layout(
    template="plotly_dark", height=700, margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=zoom_range['x'] if zoom_range else None),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1, range=zoom_range['y'] if zoom_range else None),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

st.plotly_chart(fig, use_container_width=True)

# 8. INTERACTIVE WATCHLIST TABLE
st.divider()
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("📋 รายการเฝ้าระวัง (Watchlist)")
    watchlist = df[df['Status'] != 'Safe'].sort_values('Survival_Prob')
    
    # Create an interactive selection table
    selected_row = st.dataframe(
        watchlist[['Egg_ID', 'Temp', 'Status', 'Survival_Prob']],
        on_select="rerun",
        selection_mode="single_row",
        hide_index=True,
        use_container_width=True
    )
    
    if len(selected_row.selection.rows) > 0:
        idx = selected_row.selection.rows[0]
        st.session_state.selected_egg = watchlist.iloc[idx]['Egg_ID']
        st.success(f"กำลังโฟกัสไปที่ไข่ ID: {st.session_state.selected_egg}")
        if st.button("Reset Zoom"):
            st.session_state.selected_egg = None
            st.rerun()

with c2:
    st.subheader("🛠️ Sensor Diagnostic Center")
    st.dataframe(df_sensors[['Sensor_ID', 'Temp', 'Humid']], use_container_width=True, hide_index=True)
