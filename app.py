import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# 1. ตั้งค่าหน้าเว็บให้ลื่นไหล
st.set_page_config(page_title="Duck Egg Smart Monitor", page_icon="🦆", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #daffde; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    /* ปรับแต่งปุ่มและสไลเดอร์ให้ดูทันสมัย */
    .stSlider > div [data-baseweb="slider"] { height: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦆 Smart Duck Incubator: High-Performance Twin")
st.markdown("ระบบติดตามตู้ฟักไข่เวอร์ชันปรับปรุงความเร็ว (1,000 Individual Eggs Monitoring)")

# 2. ฟังก์ชันสร้างข้อมูล 1,000 ใบ (ลดจำนวนเพื่อความลื่นไหล)
@st.cache_resource
def load_optimized_data():
    np.random.seed(42)
    # โครงสร้าง: 10 Racks x 10 Trays x 10 Eggs = 1,000 ฟอง
    racks = [f"R{i:02d}" for i in range(1, 11)]
    trays = [f"T{i:02d}" for i in range(1, 11)]
    
    egg_data = []
    
    for r_idx, r in enumerate(racks):
        for t_idx, t in enumerate(trays):
            # จำลองเซ็นเซอร์ SHT31-D ประจำถาด (100 ตัว)
            tray_temp = np.random.normal(37.5, 0.4)
            tray_humid = np.random.normal(65, 2.5)
            
            # วางไข่ 10 ฟองในถาด (เรียงแบบ 2 แถว x 5 คอลัมน์ภายในถาด)
            for e_idx in range(10):
                e_row = e_idx // 5 # 0 หรือ 1
                e_col = e_idx % 5  # 0 ถึง 4
                
                # คำนวณพิกัด X, Y สุทธิเพื่อให้เห็นระยะห่างที่สวยงาม
                final_y = (r_idx * 3) + e_row # เว้นช่องไฟระหว่าง Rack
                final_x = (t_idx * 6) + e_col # เว้นช่องไฟระหว่าง Tray
                
                egg_temp = tray_temp + np.random.normal(0, 0.05)
                egg_humid = tray_humid + np.random.normal(0, 0.2)
                spike = np.random.choice([0, 1], p=[0.94, 0.06])
                
                egg_data.append({
                    'Egg_ID': f"{r}-{t}-C{e_idx+1:02d}",
                    'Rack': r,
                    'Tray': t,
                    'Coord_X': final_x,
                    'Coord_Y': final_y,
                    'Temp': np.round(egg_temp, 2),
                    'Humid': np.round(egg_humid, 2),
                    'Spike': spike,
                    'Sensor_ID': f"SHT31-{r}-{t}"
                })
                
    df = pd.DataFrame(egg_data)
    
    # AI Training แบบรวดเร็ว
    X = df[['Temp', 'Humid', 'Spike']]
    temp_diff = np.abs(df['Temp'] - 37.5) * 20
    y = np.random.binomial(1, np.clip(100 - (temp_diff + (df['Spike']*30)), 0, 100) / 100.0)
    
    model = xgb.XGBClassifier(n_estimators=30, max_depth=3, scale_pos_weight=4)
    model.fit(X, y)
    
    return model, df

with st.spinner("🚀 กำลังเพิ่มความเร็วระบบและโหลดข้อมูล 1,000 ฟอง..."):
    model, df = load_optimized_data()

# 3. Sidebar แผงควบคุม
st.sidebar.header("⚙️ การตั้งค่าระบบ")
threshold = st.sidebar.slider("เกณฑ์แจ้งเตือนความเสี่ยง (% อัตรารอด)", 10, 90, 50)

# ทำนายความเสี่ยง
probs = model.predict_proba(df[['Temp', 'Humid', 'Spike']])[:, 1] * 100
df['Survival_Prob'] = np.round(probs, 2)

# จัดกลุ่มสีไข่ (Optimization: จัดการเป็นกลุ่มเพื่อลดจำนวน Trace ใน Plotly)
df['Risk_Category'] = 'Safe'
df.loc[df['Survival_Prob'] < threshold, 'Risk_Category'] = 'Warning'
df.loc[df['Survival_Prob'] < 35, 'Risk_Category'] = 'High Risk'

color_map = {'Safe': '#22c55e', 'Warning': '#f59e0b', 'High Risk': '#ef4444'}

# 4. แสดง Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("🥚 จำนวนไข่", "1,000")
c2.metric("🌡️ Temp Avg", f"{df['Temp'].mean():.2f} °C")
c3.metric("💧 Humid Avg", f"{df['Humid'].mean():.1f} %")
risk_num = (df['Risk_Category'] == 'High Risk').sum()
c4.metric("🚨 เสี่ยงสูง", f"{risk_num} ใบ", f"{(risk_num/10):.1f}%", delta_color="inverse")

# 5. แผนที่ดิจิทัลทวิน (Optimized Individual Map)
st.markdown("---")
st.subheader("📍 Individual Egg Map: การแสดงผลรายใบแบบ High-Speed")

fig = go.Figure()

# วาดไข่ทีละกลุ่มสีเพื่อประสิทธิภาพสูงสุด
for cat in ['Safe', 'Warning', 'High Risk']:
    sub_df = df[df['Risk_Category'] == cat]
    symbol = 'star' if cat == 'High Risk' else 'circle'
    size = 12 if cat == 'High Risk' else 9
    
    fig.add_trace(go.Scatter(
        x=sub_df['Coord_X'], y=sub_df['Coord_Y'],
        mode='markers',
        name=cat,
        marker=dict(size=size, color=color_map[cat], symbol=symbol, line=dict(width=1, color='white')),
        text=sub_df.apply(lambda r: f"ID: {r['Egg_ID']}<br>โอกาสรอด: {r['Survival_Prob']}%<br>T: {r['Temp']}°C | โซน: {r['Rack']}-{r['Tray']}", axis=1),
        hoverinfo='text'
    ))

fig.update_layout(
    template="plotly_dark",
    height=600,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
    hovermode='closest',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# 6. รายการแจ้งเตือน
if risk_num > 0:
    st.error(f"🚨 ตรวจพบไข่เสี่ยงสูง {risk_num} ใบ โปรดตรวจสอบโซนที่ทำเครื่องหมาย 'ดาวสีแดง'")
    with st.expander("🔍 ดูรายชื่อไข่ที่ต้องตรวจสอบด่วน"):
        st.table(df[df['Risk_Category'] == 'High Risk'][['Egg_ID', 'Temp', 'Humid', 'Survival_Prob', 'Sensor_ID']].sort_values('Survival_Prob').head(10))
