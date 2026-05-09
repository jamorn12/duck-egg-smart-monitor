import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Duck Egg Individual Twin", page_icon="🦆", layout="wide")

# Custom CSS เพื่อตกแต่ง Dashboard ให้ดูไฮเทค (โทน Dark Mode)
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #daffde; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    h1, h2, h3 { color: #f5f5f5; }
    div[data-testid="stExpander"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦆 Smart Duck Incubator: Individual Twin Monitor")
st.markdown("ระบบติดตามตู้ฟักไข่แบบ Digital Twin: วิเคราะห์ความเสี่ยงรายใบ (Individual Spatial Monitoring)")

# 2. ฟังก์ชันจำลองข้อมูลพร้อมพิกัดรายใบ (Coordinate Mapping Logic)
# เราจะสร้างพื้นที่ตู้ฟักขนาด 100x100 จุดพิกัด (แทนถาด 10x10 ถาด ถาดละ 10x10 ฟอง)
@st.cache_resource
def load_individual_twin_data():
    np.random.seed(42)
    num_eggs = 10000 # 10,000 ฟอง
    
    # 2.1 สร้างระบบพิกัด Grid 100x100
    racks_list = [f"R{i:02d}" for i in range(1, 11)] # R01-R10 (Row)
    trays_list = [f"T{i:02d}" for i in range(1, 11)] # T01-T10 (Col)
    
    egg_data = []
    
    # จัดการข้อมูลเซ็นเซอร์รายโซน (Zoning) - จำลองว่า 1 ถาดมี 1 เซ็นเซอร์ SHT31-D
    for r_idx, r in enumerate(racks_list):
        for t_idx, t in enumerate(trays_list):
            # จำลองค่าอุณหภูมิและความชื้นเฉลี่ยของ 'ถาด' นั้นๆ (มีความผันผวนตามโซน)
            tray_base_temp = np.random.normal(37.5, 0.5) 
            tray_base_humid = np.random.normal(65, 3.0)
            
            # วางไข่ 100 ใบลงในถาด (ใน Grid 10x10 ย่อยภายในถาด)
            for c_row in range(10): # Row ภายในถาด (0-9)
                for c_col in range(10): # Col ภายในถาด (0-9)
                    egg_idx = c_row * 10 + c_col + 1
                    
                    # ค่าเซ็นเซอร์รายใบ (มี Noise เล็กน้อย)
                    egg_temp = tray_base_temp + np.random.normal(0, 0.1)
                    egg_humid = tray_base_humid + np.random.normal(0, 0.4)
                    spike = np.random.choice([0, 1], p=[0.93, 0.07]) # ไฟตกช่วงท้าย
                    
                    # คำนวณพิกัด X, Y สุทธิสำหรับวาดแผนที่ (0-99 scale)
                    final_y = (r_idx * 10) + c_row # Rack_Number * 10 + Cell_Row
                    final_x = (t_idx * 10) + c_col # Tray_Number * 10 + Cell_Col
                    
                    egg_data.append({
                        'Egg_ID': f"{r}-{t}-C{egg_idx:03d}",
                        'Rack': r,
                        'Tray': t,
                        'Coord_X': final_x,
                        'Coord_Y': final_y,
                        'Temp': np.round(egg_temp, 2),
                        'Humid': np.round(egg_humid, 2),
                        'Spike': spike,
                        # ข้อมูลเซ็นเซอร์หลักของโซนนั้น
                        'Sensor_ID': f"SHT31-{r}-{t}", 
                        'Zone_Base_Temp': np.round(tray_base_temp, 2)
                    })
    
    df = pd.DataFrame(egg_data)
    
    # 2.2 เทรนโมเดล AI (XGBoost) บนข้อมูลรายใบ
    X = df[['Temp', 'Humid', 'Spike']]
    # จำลอง Hatch Status เพื่อให้โมเดลมีข้อมูลเรียนรู้ (ไข่อุณหภูมิสูง/ต่ำเกินไปมีโอกาสตาย)
    temp_penalty = np.abs(df['Temp'] - 37.5) * 15
    y = np.random.binomial(1, np.clip(100 - (temp_penalty + (df['Spike']*25)), 0, 100) / 100.0)
    
    model = xgb.XGBClassifier(n_estimators=50, max_depth=4, scale_pos_weight=4, random_state=42)
    model.fit(X, y)
    
    return model, df

with st.spinner("กำลังเชื่อมต่อเซ็นเซอร์ Digital Twin และประมวลผล AI รายใบ..."):
    model, df = load_individual_twin_data()

# 3. Sidebar แผงควบคุม
st.sidebar.header("⚙️ Control Panel")
threshold = st.sidebar.slider("เกณฑ์แจ้งเตือนความเสี่ยง (% โอกาสรอด)", 10, 90, 50)

# ทำนายความเสี่ยงรายใบ
probs = model.predict_proba(df[['Temp', 'Humid', 'Spike']])[:, 1] * 100
df['Survival_Prob'] = np.round(probs, 2)

# แบ่งกลุ่มความเสี่ยงเพื่อจัดสีไข่
conditions = [
    (df['Survival_Prob'] < 40), # เสี่ยงสูง
    (df['Survival_Prob'] >= 40) & (df['Survival_Prob'] < threshold), # เตือนภัย
    (df['Survival_Prob'] >= threshold) # ปลอดภัย
]
choices = ['High Risk', 'Warning', 'Safe']
df['Risk_Category'] = np.select(conditions, choices, default='Safe')

# Mapping สีตามกลุ่มความเสี่ยง (Red, Orange, Green)
color_map = {'High Risk': '#ef4444', 'Warning': '#f59e0b', 'Safe': '#22c55e'}
df['Risk_Color'] = df['Risk_Category'].map(color_map)

# 4. ส่วนแสดง Metrics รวม
col1, col2, col3, col4 = st.columns(4)
col1.metric("🥚 จำนวนไข่ทั้งหมด", f"{len(df):,} ฟอง")
col2.metric("🌡️ อุณหภูมิเฉลี่ย", f"{df['Temp'].mean():.2f} °C")
col3.metric("💧 ความชื้นเฉลี่ย", f"{df['Humid'].mean():.1f} %")
risk_count = df[df['Risk_Category'] == 'High Risk'].shape[0]
col4.metric("🚨 กลุ่มเสี่ยงสูง (High Risk)", f"{risk_count:,} ฟอง", f"{(risk_count/len(df))*100:.1f}%", delta_color="inverse")

# 5. การแสดงผล Digital Twin Map (Plotly Scatter) - ออกแบบใหม่
st.markdown("---")
st.subheader("📍 Smart Incubator Layout: Individual Egg Map (ระบุพิกัดรายใบ)")
st.info("💡 นำเมาส์ไปชี้ที่ 'จุดสี' เพื่อดูข้อมูลรายใบและข้อมูลเซ็นเซอร์โซน")

# 5.1 สร้าง Heatmap พื้นหลังแบบ Zoning เพื่อแสดงอุณหภูมิถาด (เพื่อความสวยงามและGIS)
tray_avg = df.groupby(['Rack', 'Tray', 'Sensor_ID']).agg({
    'Temp': 'mean',
    'Coord_X': 'median', # พิกัดกลางโซน
    'Coord_Y': 'median'
}).reset_index()

# 5.2 วาดจุดไข่รายใบ (Plotly Scatter)
fig = go.Figure()

# เพิ่มไข่ Safe
safe_eggs = df[df['Risk_Category'] == 'Safe']
fig.add_trace(go.Scatter(
    x=safe_eggs['Coord_X'], y=safe_eggs['Coord_Y'],
    mode='markers', name='Safe (ปกติ)',
    marker=dict(size=6, color='#22c55e', opacity=0.8, line=dict(width=0.5, color='#30363d')),
    text=safe_eggs.apply(lambda r: f"ID: {r['Egg_ID']}<br>โอกาสรอด: {r['Survival_Prob']}%<br>T: {r['Temp']}°C | H: {r['Humid']}%<br>โซน: {r['Rack']}-{r['Tray']}<br>เซ็นเซอร์: {r['Sensor_ID']}", axis=1),
    hoverinfo='text'
))

# เพิ่มไข่ Warning
warn_eggs = df[df['Risk_Category'] == 'Warning']
fig.add_trace(go.Scatter(
    x=warn_eggs['Coord_X'], y=warn_eggs['Coord_Y'],
    mode='markers', name='Warning (เฝ้าระวัง)',
    marker=dict(size=7, color='#f59e0b', opacity=0.9, symbol='circle', line=dict(width=1, color='#ffffff')),
    text=warn_eggs.apply(lambda r: f"ID: {r['Egg_ID']}<br>โอกาสรอด: {r['Survival_Prob']}%<br>T: {r['Temp']}°C | H: {r['Humid']}%<br>โซน: {r['Rack']}-{r['Tray']}", axis=1),
    hoverinfo='text'
))

# เพิ่มไข่ High Risk (ขนาดใหญ่ขึ้นและสีแดงเข้มเพื่อการแจ้งเตือน)
risk_eggs = df[df['Risk_Category'] == 'High Risk']
fig.add_trace(go.Scatter(
    x=risk_eggs['Coord_X'], y=risk_eggs['Coord_Y'],
    mode='markers', name='High Risk (เสี่ยงสูง)',
    marker=dict(size=9, color='#ef4444', opacity=1, symbol='star', line=dict(width=1.5, color='#ffffff')),
    text=risk_eggs.apply(lambda r: f"ID: {r['Egg_ID']}<br>โอกาสรอด: {r['Survival_Prob']}%<br>T: {r['Temp']}°C | H: {r['Humid']}%<br>โซน: {r['Rack']}-{r['Tray']}<br>ไฟตกปลาย: {r['Spike']}", axis=1),
    hoverinfo='text'
))

# ปรับแต่ง Layout ให้ดูเป็น GIS และ Digital Twin
fig.update_layout(
    xaxis=dict(title="Tray Location (T01-T10)", showgrid=False, zeroline=False, ticks='', showticklabels=False),
    yaxis=dict(title="Rack Location (R01-R10)", showgrid=False, zeroline=False, ticks='', showticklabels=False, scaleanchor="x", scaleratio=1),
    template="plotly_dark",
    height=800,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode='closest',
    # เพิ่มเส้นขีดแบ่ง Racks และ Trays ให้ชัดเจน
    shapes=[
        # เส้นขีดแบ่ง Trays (Col)
        dict(type="line", x0=i*10-0.5, y0=-0.5, x1=i*10-0.5, y1=99.5, line=dict(color="#30363d", width=1)) for i in range(1, 10)
    ] + [
        # เส้นขีดแบ่ง Racks (Row)
        dict(type="line", x0=-0.5, y0=i*10-0.5, x1=99.5, y1=i*10-0.5, line=dict(color="#30363d", width=1)) for i in range(1, 10)
    ]
)

st.plotly_chart(fig, use_container_width=True)

# 6. ส่วนแจ้งเตือนและ Data Explorer
st.markdown("---")
# แสดงตารางไข่ High Risk 20 อันดับแรก
alert_col, data_col = st.columns([1, 1])
with alert_col:
    st.subheader("🚨 รายการไข่เสี่ยงสูง (High Risk) 20 อันดับแรก")
    high_risk_list = df[df['Risk_Category'] == 'High Risk'].sort_values('Survival_Prob').head(20)
    if not high_risk_list.empty:
        st.dataframe(high_risk_list[['Egg_ID', 'Temp', 'Humid', 'Survival_Prob', 'Zone_Base_Temp', 'Sensor_ID']], 
                    column_config={
                        "Survival_Prob": st.column_config.NumberColumn("โอกาสรอดชีวิต (%)", format="%f"),
                        "Temp": "อุณหภูมิ (°C)",
                        "Humid": "ความชื้น (%)",
                        "Zone_Base_Temp": "อุณหภูมิโซน (°C)"
                    },
                    use_container_width=True)
    else:
        st.success("🎉 ไม่พบไข่ที่มีความเสี่ยงสูง")

with data_col:
    with st.expander("🔍 ค้นหาและตรวจสอบข้อมูลพิกัดไข่รายใบทั้งหมด"):
        st.dataframe(df[['Egg_ID', 'Temp', 'Humid', 'Survival_Prob', 'Risk_Category', 'Sensor_ID']], use_container_width=True)
