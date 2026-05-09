import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# 1. ตั้งค่าหน้าเว็บแบบ Professional
st.set_page_config(page_title="Smart Hatchery Digital Twin", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .stMetric { background-color: #161b22; border-radius: 12px; padding: 20px; border: 1px solid #30363d; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    h1, h2 { color: #f0f6fc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .status-box { padding: 10px; border-radius: 8px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏢 Duck Hatchery: Smart Factory Digital Twin")
st.markdown("ระบบบริหารจัดการโรงฟักไข่อัจฉริยะเชิงพื้นที่ (Spatial Industrial Monitoring)")

# 2. ฟังก์ชันจำลองข้อมูลและผังโรงเรือน
@st.cache_resource
def load_factory_data():
    np.random.seed(42)
    # 10 Racks x 10 Trays x 10 Eggs = 1,000 ฟอง
    racks = [f"R{i:02d}" for i in range(1, 11)]
    trays = [f"T{i:02d}" for i in range(1, 11)]
    
    egg_data = []
    sensor_data = []
    
    for r_idx, r in enumerate(racks):
        # สร้างทางเดินกลาง (Walkway) ทุกๆ 5 Racks
        y_offset = r_idx * 5 + (10 if r_idx >= 5 else 0)
        
        for t_idx, t in enumerate(trays):
            x_offset = t_idx * 8
            
            # 2.1 ข้อมูลเซ็นเซอร์ SHT31 (ติดตรงกลางถาด)
            tray_temp = np.random.normal(37.5, 0.5)
            tray_humid = np.random.normal(65, 3.0)
            sensor_data.append({
                'Sensor_ID': f"SHT31-{r}-{t}",
                'X': x_offset + 2,
                'Y': y_offset + 1,
                'Temp': np.round(tray_temp, 2),
                'Humid': np.round(tray_humid, 2)
            })
            
            # 2.2 ข้อมูลไข่ 10 ฟอง
            for e_idx in range(10):
                e_row = e_idx // 5
                e_col = e_idx % 5
                
                egg_temp = tray_temp + np.random.normal(0, 0.08)
                spike = np.random.choice([0, 1], p=[0.94, 0.06])
                
                egg_data.append({
                    'Egg_ID': f"{r}-{t}-C{e_idx+1:02d}",
                    'Rack': r, 'Tray': t,
                    'Coord_X': x_offset + e_col,
                    'Coord_Y': y_offset + e_row,
                    'Temp': np.round(egg_temp, 2),
                    'Humid': np.round(tray_humid, 2),
                    'Spike': spike,
                    'Sensor_Ref': f"SHT31-{r}-{t}"
                })
                
    df = pd.DataFrame(egg_data)
    df_sensors = pd.DataFrame(sensor_data)
    
    # AI Training
    X = df[['Temp', 'Humid', 'Spike']]
    y = np.random.binomial(1, np.clip(100 - (np.abs(df['Temp'] - 37.5) * 20 + df['Spike']*35), 0, 100) / 100.0)
    model = xgb.XGBClassifier(n_estimators=30, max_depth=3).fit(X, y)
    
    return model, df, df_sensors

model, df, df_sensors = load_factory_data()

# 3. Sidebar & Logic
st.sidebar.header("🕹️ ระบบควบคุมอาคาร")
threshold = st.sidebar.slider("เกณฑ์แจ้งเตือนความเสี่ยง (%)", 10, 90, 50)

df['Survival_Prob'] = np.round(model.predict_proba(df[['Temp', 'Humid', 'Spike']])[:, 1] * 100, 2)
df['Status'] = 'Safe'
df.loc[df['Survival_Prob'] < threshold, 'Status'] = 'Warning'
df.loc[df['Survival_Prob'] < 30, 'Status'] = 'Critical'

critical_count = (df['Status'] == 'Critical').sum()

# --- ระบบ Alert Pop-up (Toast) ---
if critical_count > 0:
    st.toast(f"🚨 ตรวจพบไข่สถานะวิกฤต {critical_count} ใบ! โปรดตรวจสอบโซนสีแดง", icon="🔥")

# 4. หน้าจอหลัก: Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("🥚 จำนวนไข่ฟัก", "1,000")
m2.metric("🌡️ อุณหภูมิเฉลี่ยอาคาร", f"{df['Temp'].mean():.2f} °C")
m3.metric("📡 เซ็นเซอร์ที่ทำงาน", f"{len(df_sensors)} จุด")
m4.metric("🚨 วิกฤต (Critical)", f"{critical_count} ใบ", delta=f"{critical_count}", delta_color="inverse")

# 5. การสร้างแผนผังอาคาร (Digital Twin Map)
st.subheader("📍 Egg Incubation Building Plan: ผังโรงเรือนและการติดตั้งเซ็นเซอร์")

fig = go.Figure()

# 5.1 วาดโครงสร้างอาคาร (กำแพงและประตู)
# กำแพงนอก
fig.add_shape(type="rect", x0=-5, y0=-5, x1=85, y1=65, line=dict(color="#8b949e", width=4))
# ทางเดินกลาง (Walkway)
fig.add_shape(type="rect", x0=-5, y0=22, x1=85, y1=32, fillcolor="rgba(139, 148, 158, 0.1)", line=dict(width=0))
# ประตู (Main Entrance)
fig.add_shape(type="line", x0=35, y0=-5, x1=45, y1=-5, line=dict(color="#58a6ff", width=8))
fig.add_annotation(x=40, y=-8, text="ทางเข้าหลัก (Entrance)", showarrow=False, font=dict(color="#58a6ff"))

# 5.2 วาดจุดไข่ (แบ่งตามสถานะ)
colors = {'Safe': '#238636', 'Warning': '#d29922', 'Critical': '#f85149'}
for status in ['Safe', 'Warning', 'Critical']:
    sub = df[df['Status'] == status]
    fig.add_trace(go.Scatter(
        x=sub['Coord_X'], y=sub['Coord_Y'], mode='markers', name=status,
        marker=dict(size=8, color=colors[status], line=dict(width=0.5, color='white')),
        text=sub.apply(lambda r: f"ID: {r['Egg_ID']}<br>โอกาสรอด: {r['Survival_Prob']}%<br>อุณหภูมิ: {r['Temp']}°C", axis=1),
        hoverinfo='text'
    ))

# 5.3 วาดจุดเซ็นเซอร์ SHT31-D (สัญลักษณ์ Diamond สีฟ้า)
fig.add_trace(go.Scatter(
    x=df_sensors['X'], y=df_sensors['Y'], mode='markers', name='Sensor Node (SHT31)',
    marker=dict(size=12, color='#1f6feb', symbol='diamond', line=dict(width=2, color='white')),
    text=df_sensors.apply(lambda r: f"<b>Sensor: {r['Sensor_ID']}</b><br>สถานะ: ออนไลน์<br>Temp: {r['Temp']}°C<br>Humid: {r['Humid']}%", axis=1),
    hoverinfo='text'
))

fig.update_layout(
    template="plotly_dark", height=700, margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

st.plotly_chart(fig, use_container_width=True)

# 6. แผงควบคุมแจ้งเตือน
c1, c2 = st.columns([1, 2])
with c1:
    st.subheader("📜 รายการเฝ้าระวัง")
    if critical_count > 0:
        st.error(f"พบจุดวิกฤตที่ต้องการการดูแลด่วน!")
        st.table(df[df['Status'] == 'Critical'][['Egg_ID', 'Temp', 'Sensor_Ref', 'Survival_Prob']].head(10))
    else:
        st.success("โรงเรือนอยู่ในสภาวะปกติ")

with c2:
    st.subheader("🛠️ สถานะเซ็นเซอร์ (Sensor Diagnostics)")
    st.dataframe(df_sensors, use_container_width=True)
