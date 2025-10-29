import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os

# ----------------------------------------------------------------------
# 0. การตั้งค่าและการโหลดฐานข้อมูล
# ----------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="AI Skincare & Makeup Advisor")

@st.cache_data
def load_db(file_path):
    """โหลดฐานข้อมูล CSV อย่างปลอดภัยและจัดการค่า Null"""
    if not os.path.exists(file_path):
        st.error(f"❗ ข้อผิดพลาดร้ายแรง: ไม่พบไฟล์ '{file_path}' กรุณาสร้างไฟล์และใส่ข้อมูล")
        return pd.DataFrame()
    
    try:
        db = pd.read_csv(file_path)
        # เตรียมข้อมูลสำหรับ Rule-Based Logic
        if 'Price_Range' in db.columns:
            db['Price_Range'] = db['Price_Range'].fillna('ไม่ระบุ')
        if 'Konvy_Link' in db.columns:
            db['Konvy_Link'] = db['Konvy_Link'].fillna('#')
        if 'Key_Ingredient' in db.columns:
             db['Key_Ingredient'] = db['Key_Ingredient'].astype(str)
        return db
    except Exception as e:
        st.error(f"❗ ข้อผิดพลาดในการอ่านไฟล์ {file_path}: {e}")
        return pd.DataFrame()

PRODUCT_DB = load_db('products.csv')
SHADE_DB = load_db('foundation_shades.csv')
TONE_DB = load_db('skin_tones.csv')


# ----------------------------------------------------------------------
# 1. ฟังก์ชันจำลองการวิเคราะห์สภาพผิวและโทนสีผิว (Simulation)
# ----------------------------------------------------------------------

def analyze_skin(uploaded_file, tone_db):
    """
    ฟังก์ชันจำลองผลการวิเคราะห์ผิวหน้าตามชื่อไฟล์ที่อัปโหลด 
    รวมถึงการจำลองโทนสีผิว (Depth, Undertone) จากฐานข้อมูล TONE_DB
    """
    if uploaded_file is None:
        return None
        
    file_name = uploaded_file.name.lower()
    
    # 1. จำลองปัญหาผิวหลัก (Skin Concern)
    if any(keyword in file_name for keyword in ["acne", "oil", "มัน", "สิว"]):
        skin_type = 'Oily'  
        acne_severity = 'Moderate'
        # จำลองโทนผิว: เน้นไปที่ Medium/Tan Warm/Neutral
        tone_options = tone_db[tone_db['Depth_Scale'].between(3.5, 6.5)]
    elif any(keyword in file_name for keyword in ["dry", "แห้ง", "sensitive", "แพ้"]):
        skin_type = 'Dry' 
        acne_severity = 'Low'
        # จำลองโทนผิว: เน้นไปที่ Fair/Light Cool/Neutral
        tone_options = tone_db[tone_db['Depth_Scale'].between(1.0, 3.5)]
    else:
        skin_type = 'Combination' 
        acne_severity = 'Low'
        # จำลองโทนผิว: เน้นไปที่ Medium Neutral
        tone_options = tone_db[tone_db['Depth_Scale'].between(3.0, 5.0)]
        
    # 2. สุ่มผลลัพธ์โทนสีผิวจากฐานข้อมูลที่กรองแล้ว
    if not tone_options.empty:
        # สุ่มเลือกแถวหนึ่งจากผลการกรอง
        random_tone = tone_options.sample(n=1).iloc[0]
        undertone = random_tone['Undertone']
        depth_scale = random_tone['Depth_Scale']
    else:
        # หากฐานข้อมูลโทนสีว่างเปล่า/กรองไม่ได้ ให้ใช้ค่า Default
        undertone = 'Neutral'
        depth_scale = 4.5
        
    return {
        'Skin_Type': skin_type,  
        'Acne_Severity': acne_severity,
        'Undertone': undertone,
        'Depth_Scale': depth_scale
    }

# ----------------------------------------------------------------------
# 2.1 ฟังก์ชันแนะนำผลิตภัณฑ์บำรุงผิว (Skincare - Rule-Based Logic)
# ----------------------------------------------------------------------

def recommend_skincare(skin_analysis_results, db):
    """กำหนดกฎเกณฑ์การแนะนำผลิตภัณฑ์บำรุงผิวตามผลการวิเคราะห์ผิว"""
    skin_type = skin_analysis_results['Skin_Type']
    acne_severity = skin_analysis_results['Acne_Severity']
    recommendations = {}

    # 1. Cleanser (ทำความสะอาด)
    if skin_type in ['Oily', 'Combination'] and acne_severity != 'Low':
        reco = db[(db['Category'] == 'Cleanser') & (db['Key_Ingredient'].str.contains('Salicylic Acid|BHA', case=False))].head(1)
    elif skin_type == 'Dry':
        reco = db[(db['Category'] == 'Cleanser') & (db['Key_Ingredient'].str.contains('Ceramide|Glycerin', case=False))].head(1)
    else:
        reco = db[db['Category'] == 'Cleanser'].head(1)
    if not reco.empty:
        recommendations['Step 1: Cleanser (ทำความสะอาด)'] = reco

    # 2. Treatment (รักษาสิว/ลดรอย)
    if acne_severity == 'Moderate':
        reco = db[(db['Key_Ingredient'].str.contains('Niacinamide|Salicylic Acid|Benzoyl Peroxide', case=False)) & (db['Category'] == 'Treatment')].head(2)
        if not reco.empty:
            recommendations['Step 2: Targeted Treatment (รักษาสิว/ลดรอย)'] = reco
    
    # 3. Moisturizer
    if skin_type in ['Oily', 'Combination']:
        reco = db[(db['Category'] == 'Moisturizer') & (db['Product_Name'].str.contains('Gel|Water|Lightweight', case=False))].head(1)
    else:
        reco = db[(db['Category'] == 'Moisturizer') & (db['Key_Ingredient'].str.contains('Ceramide|Squalane', case=False))].head(1)
    if not reco.empty:
        recommendations['Step 3: Moisturizer (เติมความชุ่มชื้น)'] = reco

    # 4. Sunscreen
    reco = db[db['Category'] == 'Sunscreen'].head(1)
    if not reco.empty:
        recommendations['Step 4: Sunscreen (ป้องกัน)'] = reco
        
    return recommendations


# ----------------------------------------------------------------------
# 2.2 ฟังก์ชันแนะนำเฉดสีรองพื้น (Foundation)
# ----------------------------------------------------------------------

def recommend_foundation(undertone, depth_scale, db):
    """จับคู่เฉดสีรองพื้นตามโทนผิวและความเข้ม"""
    if db.empty:
        return pd.DataFrame()

    filtered_df = db[db['Undertone'] == undertone]
    
    if filtered_df.empty:
        # หากไม่พบโทนที่ตรง ให้ลองใช้ Neutral
        filtered_df = db[db['Undertone'] == 'Neutral']

    if filtered_df.empty:
        return pd.DataFrame() 

    # หาค่าเฉดสีที่ใกล้เคียงกับ Depth_Scale ที่สุด
    filtered_df['Depth_Diff'] = np.abs(filtered_df['Depth_Scale'] - depth_scale)
    
    # คืนค่าเฉดสีที่ดีที่สุด 3 อันดับแรก
    return filtered_df.sort_values(by='Depth_Diff').head(3).drop(columns=['Depth_Diff']).reset_index(drop=True)


# ----------------------------------------------------------------------
# 3. Streamlit UI หลัก
# ----------------------------------------------------------------------

def main():
    if PRODUCT_DB.empty or SHADE_DB.empty or TONE_DB.empty:
        st.warning("โปรดตรวจสอบว่าไฟล์ฐานข้อมูลทั้งหมด (products.csv, foundation_shades.csv, skin_tones.csv) พร้อมใช้งาน")
        return

    st.title("🔬 AI Skincare & Makeup Advisor: Face Analysis Project")
    st.caption("ระบบจำลองการวิเคราะห์โทนสีผิว ปัญหาผิว และการแนะนำผลิตภัณฑ์")
    st.markdown("---")
    
    st.subheader("อัปโหลดรูปภาพใบหน้าของคุณ")
    st.info("💡 **เคล็ดลับ:** ลองใช้ชื่อไฟล์ที่มีคำว่า **'oil'**, **'acne'**, **'dry'** หรือ **'แห้ง'** เพื่อดูผลลัพธ์การแนะนำที่แตกต่างกัน!")
    
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        # 1. แสดงภาพและผลการวิเคราะห์
        col1, col2 = st.columns([1, 1])

        with col1:
            try:
                # แสดงภาพด้วย OpenCV เพื่อความเสถียร
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                st.image(opencv_image, channels="BGR", caption=f"ภาพที่อัปโหลด ({uploaded_file.name})", use_column_width=True)
            except Exception as e:
                st.error(f"ไม่สามารถแสดงภาพได้: {e}")

        with col2:
            st.subheader("📊 ผลการวิเคราะห์สภาพผิวและโทนสี (Simulated)")
            with st.spinner('กำลังประมวลผล...'):
                results = analyze_skin(uploaded_file, TONE_DB)
                
            if results:
                st.success("✅ วิเคราะห์สำเร็จ! (ผลลัพธ์จำลอง)")
                st.metric(label="ประเภทผิวหลัก", value=f"**{results['Skin_Type']}**")
                st.metric(label="ระดับความรุนแรงของสิว", value=f"**{results['Acne_Severity']}**")
                st.info(f"**โทนสีผิว (Undertone):** {results['Undertone']} | **ระดับความเข้ม (Depth):** {results['Depth_Scale']:.1f}")

        st.markdown("---")
        
        # 2. แสดงผลการแนะนำ Skincare
        st.header("🧴 ผลิตภัณฑ์บำรุงผิวที่แนะนำ (Skincare)")
        skincare_recommendations = recommend_skincare(results, PRODUCT_DB)

        if not skincare_recommendations:
            st.warning("ไม่พบผลิตภัณฑ์บำรุงผิวที่ตรงกับเงื่อนไขในฐานข้อมูล 'products.csv'.")
        else:
            for category, df_reco in skincare_recommendations.items():
                st.markdown(f"#### {category}")
                for _, row in df_reco.iterrows():
                    st.markdown(f"**{row['Product_Name']}** (แบรนด์: {row['Brand']})")
                    st.markdown(f"**จุดเด่น:** *{row['Key_Ingredient']}* | ราคา: {row['Price_Range']}")
                    st.markdown(f"[🔗 ดูรายละเอียดสินค้า]({row['Konvy_Link']})")
                    st.markdown("---")
        
        st.markdown("---")

        # 3. แสดงผลการแนะนำ Foundation
        st.header("🎨 เฉดสีรองพื้นที่แนะนำ (Foundation)")
        
        foundation_recommendations = recommend_foundation(
            results['Undertone'], results['Depth_Scale'], SHADE_DB
        )

        if not foundation_recommendations.empty:
            st.markdown(f"**เฉดสีที่ใกล้เคียงที่สุด** สำหรับโทน **{results['Undertone']}** และผิวระดับ **{results['Depth_Scale']:.1f}**:")
            st.dataframe(
                foundation_recommendations[['Shade_Name', 'Brand', 'Undertone', 'Depth_Scale', 'Coverage', 'Konvy_Link']],
                use_container_width=True
            )
        else:
            st.warning("ไม่พบเฉดสีที่ใกล้เคียงในฐานข้อมูล 'foundation_shades.csv'.")


if __name__ == "__main__":
    main()
