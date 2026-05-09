import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Duck Egg Digital Twin", page_icon="🦆", layout="wide")

# Custom CSS เพื่อตกแต่งให้เหมือนแดชบอร์ดล้ำๆ
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_stdio=True)

st.title("🦆 Duck Incubator Digital Twin")
st.markdown("ระบบควบคุมและติดตามตู้ฟักไข่อัจฉริยะ (Real-time Spatial Monitoring)")

# 2. ฟังก์ชันจำลองข้อมูลพร้อมพิกัด (Grid System)
@st.cache_resource
def load_advanced_data():
    np.random.seed(42)
    num_eggs = 10000 # จำลอง 10,000 ฟองตามโจทย์
    
    # สุ่มข้อมูลพื้นฐาน
    avg_temp = np.random.normal(37.5, 0.8, num_eggs)
    avg_humidity = np.random.normal(65, 5.0, num_eggs)
    late_stage_spike = np.random.choice([0, 1], p=[0.90, 0.10], size=num_eggs)
    
    # สร้างระบบพิกัด (Rack R01-R10, Tray T01-T10)
    racks = [f"R{i:02d}" for i in range(1, 11)]
    trays = [f"T{i:02d}" for i in range(1, 11)]
    
    egg_data = []
    egg_idx = 0
    for r in racks:
        for t in trays:
            for c in range(1, 101): # 100 ฟองต่อถาด
                egg_data.append({
                    'Egg_ID': f"{r}-{t}-C{c:03d}",
                    'Rack': r,
                    'Tray': t,
                    'Row': int(r[1:]) - 1, # พิกัดสำหรับวาดกราฟ
                    'Col': int(t[1:]) - 1,
                    'Temp': np.round(avg_temp[egg_idx], 2),
                    'Humid': np.round(avg_humidity[egg_idx], 2),
                    'Spike': late_stage_spike[egg_idx]
                })
                egg_idx += 1
                if egg_idx >= num_eggs: break
    
    df = pd.DataFrame(egg_data)
    
    # คำนวณความเสี่ยงด้วย Logic พื้นฐานก่อน (เพื่อสร้าง Label สำหรับเทรน)
    temp_p = np.abs(df['Temp'] - 37.5) * 15
    late_p = np.where(df['Spike'] == 1, 30, 0)
    df['Status'] = np.random.binomial(1, np.clip(100 - (temp_p + late_p), 0, 100) / 100.0)
    
    # เทรนโมเดล XGBoost
    X = df[['Temp', 'Humid', 'Spike']]
    y = df['Status']
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=4, random_state=42)
    model.fit(X, y)
    
    return model, df

model, df = load_advanced_data()

# 3. Sidebar ตั้งค่า
st.sidebar.header("🕹️ Control Panel")
threshold = st.sidebar.slider("เกณฑ์แจ้งเตือนความเสี่ยง (%)", 10, 90, 50)

# ทำนายผล
probs = model.predict_proba(df[['Temp', 'Humid', 'Spike']])[:, 1] * 100
df['Survival_Prob'] = np.round(probs, 2)
df['Is_At_Risk'] = df['Survival_Prob'] < threshold

# 4. ส่วนแสดงผล Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("🥚 จำนวนไข่ทั้งหมด", f"{len(df):,} ฟอง")
col2.metric("🌡️ อุณหภูมิเฉลี่ย", f"{df['Temp'].mean():.2f} °C")
col3.metric("💧 ความชื้นเฉลี่ย", f"{df['Humid'].mean():.1f} %")
at_risk_count = df['Is_At_Risk'].sum()
col4.metric("🚨 ไข่ที่เสี่ยงสูง", f"{at_risk_count:,} ฟอง", f"{(at_risk_count/len(df))*100:.1f}%", delta_color="inverse")

# 5. แผนที่ไข่ (Digital Twin Map) - Zoning 10x10
st.markdown("---")
st.subheader("📍 Farm Layout: แผนที่ความเสี่ยงรายถาด (Zoning Map)")

# สรุปข้อมูลรายถาด (Zoning) เพื่อวาดแผนที่
tray_map = df.groupby(['Row', 'Col', 'Rack', 'Tray']).agg({
    'Temp': 'mean',
    'Humid': 'mean',
    'Is_At_Risk': 'sum', # จำนวนไข่ที่เสี่ยงในถาดนั้น
    'Survival_Prob': 'mean'
}).reset_index()

# สร้าง Heatmap ด้วย Plotly
fig = go.Figure(data=go.Heatmap(
    z=tray_map['Is_At_Risk'],
    x=tray_map['Tray'],
    y=tray_map['Rack'],
    colorscale='YlOrRd',
    text=tray_map.apply(lambda r: f"Rack: {r['Rack']}<br>Tray: {r['Tray']}<br>Avg Temp: {r['Temp']:.2f}°C<br>At Risk: {r['Is_At_Risk']} eggs", axis=1),
    hoverinfo='text'
))

fig.update_layout(
    title="Heatmap แสดงจำนวนไข่ที่มีความเสี่ยงต่อถาด (Risk per Tray)",
    xaxis_title="Tray Number",
    yaxis_title="Rack Number",
    template="plotly_dark",
    height=600
)

# แสดงแผนที่และตารางแจ้งเตือน
map_col, table_col = st.columns([2, 1])

with map_col:
    st.plotly_chart(fig, use_container_width=True)

with table_col:
    st.subheader("🚨 รายการแจ้งเตือนด่วน")
    critical_trays = tray_map[tray_map['Is_At_Risk'] > 10].sort_values('Is_At_Risk', ascending=False)
    if not critical_trays.empty:
        for _, row in critical_trays.head(5).iterrows():
            st.error(f"**{row['Rack']}-{row['Tray']}**: พบไข่เสี่ยง {row['Is_At_Risk']} ฟอง")
    else:
        st.success("ทุกโซนอยู่ในสภาวะปกติ")

# 6. รายการไข่รายใบ (Data Explorer)
with st.expander("🔍 ดูข้อมูลพิกัดไข่รายใบทั้งหมด"):
    st.dataframe(df[['Egg_ID', 'Temp', 'Humid', 'Survival_Prob', 'Is_At_Risk']].sort_values('Survival_Prob'), use_container_width=True)
