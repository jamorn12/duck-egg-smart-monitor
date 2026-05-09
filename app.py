import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Duck Egg Smart Monitor", page_icon="🦆", layout="wide")
st.title("🦆 ระบบติดตามตู้ฟักไข่อัจฉริยะ (AI Monitor)")
st.markdown("ระบบวิเคราะห์ความเสี่ยงไข่ตายโคมด้วย Machine Learning (XGBoost)")

# 2. ฟังก์ชันจำลองข้อมูลและเทรนโมเดล (Cache ไว้จะได้ไม่หน่วงเว็บ)
@st.cache_resource
def load_model_and_data():
    np.random.seed(42)
    num_eggs = 20000
    
    # สร้างข้อมูลจำลอง
    avg_temp = np.random.normal(37.5, 0.8, num_eggs)
    avg_humidity = np.random.normal(65, 5.0, num_eggs)
    late_stage_spike = np.random.choice([0, 1], p=[0.85, 0.15], size=num_eggs)
    
    temp_penalty = np.abs(avg_temp - 37.5) * 15
    humid_penalty = np.abs(avg_humidity - 65) * 1.5
    late_penalty = np.where(late_stage_spike == 1, 30, 0)
    
    traditional_score = np.clip(100 - (temp_penalty + humid_penalty + late_penalty), 0, 100)
    hatch_status = np.random.binomial(1, traditional_score / 100.0)
    
    df = pd.DataFrame({
        'Egg_ID': range(1, num_eggs + 1),
        'Avg_Temperature_C': np.round(avg_temp, 2),
        'Avg_Humidity_Pct': np.round(avg_humidity, 2),
        'Late_Stage_Spike': late_stage_spike,
        'Hatch_Status_Actual': hatch_status
    })
    
    # เทรนโมเดล XGBoost แบบถ่วงน้ำหนัก (Class Imbalance)
    X = df[['Avg_Temperature_C', 'Avg_Humidity_Pct', 'Late_Stage_Spike']]
    y = df['Hatch_Status_Actual']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ถ่วงน้ำหนักให้คลาส 0 (ตายโคม) สำคัญขึ้น
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, scale_pos_weight=scale_pos_weight, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# โหลดโมเดล
with st.spinner("กำลังเชื่อมต่อเซิร์ฟเวอร์และประมวลผล AI..."):
    model, X_test, y_test = load_model_and_data()

# 3. แถบตั้งค่าด้านข้าง
st.sidebar.header("⚙️ การตั้งค่าระบบ")
threshold = st.sidebar.slider("เกณฑ์แจ้งเตือนความเสี่ยง (%)", min_value=10, max_value=90, value=50, help="หากโอกาสรอดชีวิตต่ำกว่าเกณฑ์นี้ ระบบจะแจ้งเตือนทันที")

# 4. ประมวลผลและแสดงตัวเลข
probs = model.predict_proba(X_test)[:, 1] * 100 # โอกาสรอด (%)
results_df = X_test.copy()
results_df['Survival_Probability'] = np.round(probs, 2)
results_df['Alert'] = np.where(probs < threshold, "🚨 เสี่ยงตายโคม", "✅ ปกติ")

at_risk_eggs = results_df[results_df['Alert'] == "🚨 เสี่ยงตายโคม"].sort_values('Survival_Probability')
safe_count = len(results_df) - len(at_risk_eggs)

col1, col2, col3 = st.columns(3)
col1.metric("🥚 จำนวนไข่ที่เฝ้าระวังทั้งหมด", f"{len(results_df):,} ฟอง")
col2.metric("✅ ไข่ปกติ (Safe)", f"{safe_count:,} ฟอง")
col3.metric("🚨 เสี่ยงตายโคม (At Risk)", f"{len(at_risk_eggs):,} ฟอง", "- ต้องการตรวจสอบด่วน", delta_color="inverse")

# 5. แสดงตารางข้อมูล
st.markdown("---")
st.subheader(f"📋 รายการไข่ที่ต้องตรวจสอบด่วน (โอกาสรอดต่ำกว่า {threshold}%)")

if len(at_risk_eggs) > 0:
    st.dataframe(
        at_risk_eggs[['Avg_Temperature_C', 'Avg_Humidity_Pct', 'Late_Stage_Spike', 'Survival_Probability', 'Alert']],
        column_config={
            "Avg_Temperature_C": "อุณหภูมิเฉลี่ย (°C)",
            "Avg_Humidity_Pct": "ความชื้นเฉลี่ย (%)",
            "Late_Stage_Spike": "มีไฟตกช่วงท้าย (1=มี)",
            "Survival_Probability": st.column_config.ProgressColumn(
                "โอกาสรอดชีวิต (%)", format="%f", min_value=0, max_value=100
            ),
        },
        use_container_width=True
    )
else:
    st.success("🎉 ไม่พบไข่ที่มีความเสี่ยงในเกณฑ์ที่กำหนด")
