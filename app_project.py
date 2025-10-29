import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os

# ----------------------------------------------------------------------
# 0. การตั้งค่าและการโหลดฐานข้อมูล
# ----------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="AI Skincare Advisor Project")

@st.cache_data
def load_product_db():
    """โหลดฐานข้อมูลผลิตภัณฑ์จาก products.csv อย่างปลอดภัย"""
    file_path = 'products.csv'
    if not os.path.exists(file_path):
        st.error(f"❗ ข้อผิดพลาดร้ายแรง: ไม่พบไฟล์ '{file_path}' กรุณาสร้างไฟล์และใส่ข้อมูล")
        return pd.DataFrame()
    
    try:
        db = pd.read_csv(file_path)
        # เตรียมข้อมูลสำหรับ Rule-Based Logic
        db['Price_Range'] = db['Price_Range'].fillna('ไม่ระบุ')
        db['Konvy_Link'] = db['Konvy_Link'].fillna('#')
        # แปลง Key_Ingredient เป็นสตริงสำหรับใช้ str.contains อย่างปลอดภัย
        db['Key_Ingredient'] = db['Key_Ingredient'].astype(str) 
        return db
    except Exception as e:
        st.error(f"❗ ข้อผิดพลาดในการอ่านไฟล์ CSV: {e}")
        return pd.DataFrame()

PRODUCT_DB = load_product_db()

# ----------------------------------------------------------------------
# 1. ฟังก์ชันจำลองการวิเคราะห์สภาพผิว (Simulation)
# ----------------------------------------------------------------------

def analyze_skin(uploaded_file):
    """
    ฟังก์ชันจำลองผลการวิเคราะห์ผิวหน้าตามชื่อไฟล์ที่อัปโหลด
    """
    if uploaded_file is None:
        return None
        
    file_name = uploaded_file.name.lower()
    
    # กฎการจำลอง 1: ผิวมีปัญหา (Oily/Acne-Prone)
    if any(keyword in file_name for keyword in ["acne", "oil", "มัน", "สิว"]):
        return {
            'Skin_Type': 'Oily',  
            'Acne_Severity': 'Moderate',
            'Oiliness_Score': 0.85
        }
    # กฎการจำลอง 2: ผิวไม่มีปัญหา (Combination/Normal)
    elif any(keyword in file_name for keyword in ["normal", "dry", "combination", "ปกติ", "แห้ง"]):
        return {
            'Skin_Type': 'Dry', 
            'Acne_Severity': 'Low',
            'Oiliness_Score': 0.30
        }
    # กฎการจำลอง 3: ค่าเริ่มต้น
    else:
        return {
            'Skin_Type': 'Combination', 
            'Acne_Severity': 'Low',
            'Oiliness_Score': 0.45
        }


# ----------------------------------------------------------------------
# 2. ฟังก์ชันแนะนำผลิตภัณฑ์ (Rule-Based Logic)
# ----------------------------------------------------------------------

def recommend_products(skin_analysis_results, db):
    """กำหนดกฎเกณฑ์การแนะนำผลิตภัณฑ์ตามผลการวิเคราะห์ผิว"""
    skin_type = skin_analysis_results['Skin_Type']
    acne_severity = skin_analysis_results['Acne_Severity']
    recommendations = {}

    # 1. Cleanser (ทำความสะอาด)
    if skin_type in ['Oily', 'Combination'] and acne_severity != 'Low':
        reco = db[(db['Category'] == 'Cleanser') & (db['Key_Ingredient'].str.contains('Salicylic Acid|BHA', case=False))].head(1)
    elif skin_type == 'Dry':
        reco = db[(db['Category'] == 'Cleanser') & (db['Key_Ingredient'].str.contains('Ceramide|Glycerin', case=False))].head(1)
    else:
        reco = db[db['Category'] == 'Cleanser'].head(1) # คลีนเซอร์ทั่วไป
        
    if not reco.empty:
        recommendations['Step 1: Cleanser (ทำความสะอาด)'] = reco

    # 2. Treatment (รักษาสิว/ลดรอย)
    if acne_severity == 'Moderate':
        reco = db[(db['Key_Ingredient'].str.contains('Niacinamide|Salicylic Acid|Benzoyl Peroxide', case=False)) & (db['Category'] == 'Treatment')].head(2)
        if not reco.empty:
            recommendations['Step 2: Targeted Treatment (รักษาสิว/ลดรอย)'] = reco
    
    # 3. Serum/Toner (เพิ่มความชุ่มชื้น/ลดความมัน)
    if skin_type in ['Oily', 'Combination']:
        reco = db[(db['Category'].isin(['Serum', 'Toner'])) & (db['Key_Ingredient'].str.contains('Niacinamide|Green Tea', case=False))].head(1)
    elif skin_type == 'Dry':
        reco = db[(db['Category'] == 'Serum') & (db['Key_Ingredient'].str.contains('Hyaluronic Acid|Ceramide', case=False))].head(1)
    
    if not reco.empty:
        recommendations['Step 3: Serum/Essence (บำรุงเฉพาะจุด)'] = reco
        

    # 4. Moisturizer (เติมความชุ่มชื้น)
    if skin_type in ['Oily', 'Combination']:
        reco = db[(db['Category'] == 'Moisturizer') & (db['Product_Name'].str.contains('Gel|Water|Lightweight', case=False))].head(1)
    else:
        reco = db[(db['Category'] == 'Moisturizer') & (db['Key_Ingredient'].str.contains('Ceramide|Squalane', case=False))].head(1)
    
    if not reco.empty:
        recommendations['Step 4: Moisturizer (เติมความชุ่มชื้น)'] = reco

    # 5. Sunscreen (ป้องกัน)
    reco = db[db['Category'] == 'Sunscreen'].head(1)
    if not reco.empty:
        recommendations['Step 5: Sunscreen (ป้องกัน)'] = reco
        
    return recommendations

# ----------------------------------------------------------------------
# 3. Streamlit UI หลัก
# ----------------------------------------------------------------------

def main():
    if PRODUCT_DB.empty:
        st.warning("โปรดตรวจสอบไฟล์ 'products.csv' ก่อนดำเนินการต่อ")
        return

    st.title("🔬 AI Skincare Advisor: Face Analysis Project")
    st.caption("ระบบจำลองการวิเคราะห์สภาพผิวและการแนะนำผลิตภัณฑ์ (สำหรับงานนำเสนอ)")
    st.markdown("---")
    
    st.subheader("อัปโหลดรูปภาพใบหน้าของคุณ")
    st.info("💡 **เคล็ดลับ:** ลองใช้ชื่อไฟล์ที่มีคำว่า **'oil'**, **'acne'** หรือ **'แห้ง'** เพื่อดูผลลัพธ์การแนะนำที่แตกต่างกัน!")
    
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        # 1. แสดงภาพและผลการวิเคราะห์
        col1, col2 = st.columns([1, 1])

        with col1:
            try:
                # การอ่านไฟล์ด้วย numpy และ cv2 นั้นเสถียรสำหรับ Streamlit
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                st.image(opencv_image, channels="BGR", caption=f"ภาพที่อัปโหลด ({uploaded_file.name})", use_column_width=True)
            except Exception as e:
                st.error(f"ไม่สามารถแสดงภาพได้: {e}")

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
        
        # 2. แสดงผลการแนะนำ
        st.header("🛒 ผลิตภัณฑ์แนะนำสำหรับผิวคุณ")
        product_recommendations = recommend_products(results, PRODUCT_DB)

        if not product_recommendations:
            st.warning("ไม่พบผลิตภัณฑ์ที่ตรงกับเงื่อนไขในฐานข้อมูล 'products.csv'.")
            return

        # แสดงผลลัพธ์การแนะนำ
        for category, df_reco in product_recommendations.items():
            st.markdown(f"#### {category}")
            
            for _, row in df_reco.iterrows():
                st.markdown(f"**{row['Product_Name']}** (แบรนด์: {row['Brand']})")
                st.markdown(f"**จุดเด่น:** *{row['Key_Ingredient']}* | ราคา: {row['Price_Range']}")
                st.markdown(f"ประเภทผิว: {row['Target_Skin_Type']} | หมวดหมู่: {row['Category']}")
                st.markdown(f"[🔗 ดูรายละเอียดสินค้า]({row['Konvy_Link']})")
                st.markdown("---")

if __name__ == "__main__":
    main()
