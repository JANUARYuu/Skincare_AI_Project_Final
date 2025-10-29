import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os

# --- 0. การตั้งค่าและการโหลดฐานข้อมูล ---

st.set_page_config(layout="wide", page_title="AI Skincare Advisor Project")

@st.cache_data
def load_product_db():
    """โหลดฐานข้อมูลผลิตภัณฑ์จาก products.csv"""
    file_path = 'products.csv'
    if not os.path.exists(file_path):
        st.error(f"❗ ข้อผิดพลาดร้ายแรง: ไม่พบไฟล์ '{file_path}' กรุณาสร้างไฟล์และใส่ข้อมูลตามที่แนะนำ")
        return pd.DataFrame()
    
    try:
        db = pd.read_csv(file_path)
        db['Price_Range'] = db['Price_Range'].fillna('ไม่ระบุ')
        db['Konvy_Link'] = db['Konvy_Link'].fillna('#')
        return db
    except Exception as e:
        st.error(f"❗ ข้อผิดพลาดในการอ่านไฟล์ CSV: {e}")
        return pd.DataFrame()

PRODUCT_DB = load_product_db()

# --- 1. ฟังก์ชันจำลองการวิเคราะห์สภาพผิว (Machine Learning Mock) ---

def analyze_skin(uploaded_file):
    """
    ฟังก์ชันจำลองผลการวิเคราะห์ผิวหน้าตามชื่อไฟล์ที่อัปโหลด
    (เพื่อสาธิตว่าผลลัพธ์ต่างกันแล้วแนะนำต่างกัน)
    """
    if uploaded_file is None:
        return None
        
    file_name = uploaded_file.name.lower()
    
    if "acne" in file_name or "oil" in file_name or "มัน" in file_name or "สิว" in file_name:
        # กรณีผิวมีปัญหา: หน้ามัน มีสิวปานกลาง
        return {
            'Skin_Type': 'Oily',  
            'Acne_Severity': 'Moderate',
            'Oiliness_Score': 0.85
        }
    else:
        # กรณีผิวปกติ: ผิวผสม ไม่มีสิว
        return {
            'Skin_Type': 'Combination', 
            'Acne_Severity': 'Low',
            'Oiliness_Score': 0.45
        }

# --- 2. ฟังก์ชันแนะนำผลิตภัณฑ์ (Rule-Based Logic) ---

def recommend_products(skin_analysis_results, db):
    """กำหนดกฎเกณฑ์การแนะนำผลิตภัณฑ์ตามผลการวิเคราะห์ผิว"""
    skin_type = skin_analysis_results['Skin_Type']
    acne_severity = skin_analysis_results['Acne_Severity']
    recommendations = {}

    # 1. Cleanser
    if skin_type in ['Oily', 'Combination'] and acne_severity != 'Low':
        reco = db[(db['Category'] == 'Cleanser') & (db['Key_Ingredient'].str.contains('Salicylic Acid', case=False, na=False))].head(1)
    else:
        reco = db[db['Category'] == 'Cleanser'].head(1)
        
    if not reco.empty:
        recommendations['Step 1: Cleanser (ทำความสะอาด)'] = reco.iloc[0]

    # 2. Treatment (สำคัญสำหรับสิว)
    if acne_severity == 'Moderate':
        reco = db[db['Key_Ingredient'].str.contains('Benzoyl Peroxide|Niacinamide', case=False, na=False)].head(2)
        if not reco.empty:
            recommendations['Step 2: Targeted Treatment (รักษาสิว/ลดรอย)'] = reco

    # 3. Moisturizer (เนื้อบางเบาสำหรับหน้ามัน)
    if skin_type in ['Oily', 'Combination']:
        reco = db[(db['Category'] == 'Moisturizer') & (db['Product_Name'].str.contains('Gel|Water', case=False, na=False))].head(1)
    else:
        reco = db[db['Category'] == 'Moisturizer'].head(1) # มอยส์เจอไรเซอร์ทั่วไป
    
    if not reco.empty:
        recommendations['Step 3: Moisturizer (เติมความชุ่มชื้น)'] = reco.iloc[0]

    # 4. Sunscreen
    reco = db[db['Category'] == 'Sunscreen'].head(1)
    if not reco.empty:
        recommendations['Step 4: Sunscreen (ป้องกัน)'] = reco.iloc[0]
        
    return recommendations

# --- 3. Streamlit UI หลัก ---

def main():
    if PRODUCT_DB.empty:
        st.warning("โปรดแก้ไขไฟล์ 'products.csv' ก่อนดำเนินการต่อ")
        return

    st.title("🔬 AI Skincare Advisor: Face Analysis Project")
    st.caption("ระบบจำลองการวิเคราะห์สภาพผิวและการแนะนำผลิตภัณฑ์ (สำหรับงานนำเสนอ)")
    st.markdown("---")
    
    st.subheader("อัปโหลดรูปภาพใบหน้าของคุณ")
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ (JPG/PNG) - ลองใช้ชื่อไฟล์ที่มีคำว่า 'oil' หรือ 'acne' เพื่อดูผลลัพธ์สำหรับผิวมีปัญหา", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            # แสดงรูปภาพ
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            st.image(opencv_image, channels="BGR", caption=f"ภาพที่อัปโหลด ({uploaded_file.name})", use_column_width=True)

        with col2:
            st.subheader("📊 ผลการวิเคราะห์สภาพผิว (Simulated)")
            with st.spinner('กำลังประมวลผล...'):
                results = analyze_skin(uploaded_file)
                
            if results:
                st.success("✅ วิเคราะห์สำเร็จ! (ผลลัพธ์จำลอง)")
                st.metric(label="ประเภทผิวหลัก", value=f"**{results['Skin_Type']}**")
                st.metric(label="ระดับความรุนแรงของสิว", value=f"**{results['Acne_Severity']}**")
                st.caption(f"คะแนนความมัน: {results['Oiliness_Score']:.2f}")

                st.markdown("---")
                st.header("🛒 ผลิตภัณฑ์แนะนำสำหรับผิวคุณ")
                product_recommendations = recommend_products(results, PRODUCT_DB)

                if not product_recommendations:
                    st.warning("ไม่พบผลิตภัณฑ์ที่ตรงกับเงื่อนไขในฐานข้อมูล 'products.csv'.")
                    return

                # แสดงผลลัพธ์การแนะนำ
                for category, data in product_recommendations.items():
                    st.markdown(f"#### {category}")
                    
                    if isinstance(data, pd.Series):
                        df = pd.DataFrame([data])
                    else:
                        df = data
                        
                    for _, row in df.iterrows():
                        st.markdown(f"**{row['Product_Name']}** (แบรนด์: {row['Brand']})")
                        st.markdown(f"**จุดเด่น:** *{row['Key_Ingredient']}* | ราคา: {row['Price_Range']}")
                        st.markdown(f"[🔗 ดูรายละเอียดสินค้า (ลิงก์จำลอง)]({row['Konvy_Link']})")
                        st.markdown("---")

if __name__ == "__main__":
    main()
