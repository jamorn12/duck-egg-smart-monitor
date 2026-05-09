import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Duck Egg Digital Twin", page_icon="🦆", layout="wide")

# Custom CSS ตกแต่ง Dashboard (แก้ไขพารามิเตอร์ที่ผิดให้แล้วครับ!)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True) # <-- แก้ไขจาก unsafe_allow_stdio เป็น unsafe_allow_html

st.title("🦆 Duck Incubator Digital Twin")
st.markdown("ระบบควบคุมและติดตามตู้ฟักไข่อัจฉริยะ (Real-time Spatial Monitoring)")

# 2. ฟังก์ชันจำลองข้อมูลพร้อมพิกัด (Grid System)
@st.cache_resource
def load_advanced_data():
    np.random.seed(42)
    num_eggs = 10000 
    
    racks = [f"R{i:02d}" for i in range(1, 11)]
    trays = [f"T{i:02d}" for i in range(1, 11)]
    
    egg_data = []
    idx = 0
    for r in racks:
        for t in trays:
            # สุ่มค่าเซ็นเซอร์รายถาด (Zoning)
            tray_temp = np.random.normal(37.5, 0.4)
            tray_humid = np.random.normal(65, 2.0)
            for c in range(1, 101):
                egg_temp = tray_temp + np.random.normal(0, 0.1)
                egg_humid = tray_humid + np.random.normal(0, 0.5)
                spike = np.random.choice([0, 1], p=[0.95, 0.05])
                
                egg_data.append({
                    'Egg_ID': f"{r}-{t}-C{c:03d}",
                    'Rack': r,
                    'Tray': t,
                    'Row': int(r[1:]) - 1,
                    'Col': int(t[1:]) - 1,
                    'Temp': np.round(egg_temp, 2),
                    'Humid': np.round(egg_humid, 2),
                    'Spike': spike
                })
                idx += 1
    
    df = pd.DataFrame(egg_data)
    
    # AI Logic สำหรับทำนาย
    X = df[['Temp', 'Humid', 'Spike']]
    # จำลอง Hatch Status เพื่อให้โมเดลมีข้อมูลเรียนรู้
    temp_p = np.abs(df['Temp'] - 37.5) * 15
    y = np.random.binomial(1, np.clip(100 - (temp_p + (df['Spike']*20)), 0, 100) / 100.0)
    
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, scale_pos_weight=5, random_state=42)
    model.fit(X, y)
    
    return model, df

with st.spinner("กำลังเชื่อมต่อเซ็นเซอร์ Digital Twin..."):
    model, df = load_advanced_data()

# 3. Sidebar
st.sidebar.header("🕹️ Control Panel")
threshold = st.sidebar.slider("เกณฑ์แจ้งเตือนความเสี่ยง (%)", 10, 90, 50)

# ทำนาย
probs = model.predict_proba(df[['Temp', 'Humid', 'Spike']])[:, 1] * 100
df['Survival_Prob'] = np.round(probs, 2)
df['Is_At_Risk'] = df['Survival_Prob'] < threshold

# 4. Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("🥚 ไข่ทั้งหมด", f"{len(df):,} ฟอง")
col2.metric("🌡️ อุณหภูมิเฉลี่ย", f"{df['Temp'].mean():.2f} °C")
col3.metric("💧 ความชื้นเฉลี่ย", f"{df['Humid'].mean():.1f} %")
risk_count = df['Is_At_Risk'].sum()
col4.metric("🚨 ไข่เสี่ยงสูง", f"{risk_count:,} ฟอง", f"{(risk_count/len(df))*100:.1f}%", delta_color="inverse")

# 5. Digital Twin Map (Plotly)
st.markdown("---")
st.subheader("📍 Real-time Monitoring Map (Zoning)")

tray_summary = df.groupby(['Row', 'Col', 'Rack', 'Tray']).agg({
    'Temp': 'mean',
    'Humid': 'mean',
    'Is_At_Risk': 'sum'
}).reset_index()

fig = go.Figure(data=go.Heatmap(
    z=tray_summary['Is_At_Risk'],
    x=tray_summary['Tray'],
    y=tray_summary['Rack'],
    colorscale='YlOrRd',
    text=tray_summary.apply(lambda r: f"โซน: {r['Rack']}-{r['Tray']}<br>อุณหภูมิ: {r['Temp']:.2f}°C<br>พบความเสี่ยง: {r['Is_At_Risk']} ใบ", axis=1),
    hoverinfo='text'
))

fig.update_layout(
    xaxis_title="Tray Number",
    yaxis_title="Rack Number",
    template="plotly_dark",
    height=600
)

m_col, t_col = st.columns([2, 1])
with m_col:
    st.plotly_chart(fig, use_container_width=True)
with t_col:
    st.subheader("⚠️ แจ้งเตือนโซนวิกฤต")
    bad_zones = tray_summary[tray_summary['Is_At_Risk'] > 5].sort_values('Is_At_Risk', ascending=False)
    if not bad_zones.empty:
        for _, r in bad_zones.head(5).iterrows():
            st.error(f"โซน **{r['Rack']}-{r['Tray']}**: เสี่ยง {r['Is_At_Risk']} ใบ")
    else:
        st.success("สภาพอากาศในทุกโซนปกติ")

with st.expander("🔍 ตรวจสอบพิกัดไข่รายใบ (Data Table)"):
    st.dataframe(df[['Egg_ID', 'Temp', 'Humid', 'Survival_Prob', 'Is_At_Risk']].sort_values('Survival_Prob'), use_container_width=True)
