import streamlit as st
import pandas as pd
import numpy as np
# Note: หากคุณใช้ OpenCV ในส่วนอื่นของโค้ดโปรดเพิ่ม 'import cv2' ที่นี่
# import cv2 

# ----------------------------------------------------------------------
# 1. การโหลดฐานข้อมูล (ต้องไม่มี 'FileNotFoundError' ใน Streamlit Cloud)
# ----------------------------------------------------------------------

# โหลดฐานข้อมูลผลิตภัณฑ์บำรุงผิว
try:
    PRODUCT_DB = pd.read_csv("products.csv")
except FileNotFoundError:
    st.error("❌ ERROR: ไม่พบไฟล์ 'products.csv' โปรดตรวจสอบใน GitHub.")
    PRODUCT_DB = pd.DataFrame() 

# โหลดฐานข้อมูลเฉดสีรองพื้น
try:
    SHADE_DB = pd.read_csv("foundation_shades.csv")
except FileNotFoundError:
    st.warning("⚠️ Warning: ไม่พบไฟล์ 'foundation_shades.csv' ฟังก์ชันแนะนำรองพื้นจะถูกจำกัด")
    SHADE_DB = pd.DataFrame() 


# ----------------------------------------------------------------------
# 2. ฟังก์ชันแนะนำผลิตภัณฑ์บำรุงผิว
# ----------------------------------------------------------------------

def recommend_skincare(skin_type, max_price, category):
    """ฟังก์ชันแนะนำผลิตภัณฑ์บำรุงผิวตามประเภทผิว, ราคา และหมวดหมู่"""
    if PRODUCT_DB.empty:
        return pd.DataFrame()
    
    df = PRODUCT_DB.copy()
    
    # แปลง Price_Range เป็นตัวเลขเพื่อเปรียบเทียบราคา
    # สมมติฐาน: Low=1, Medium=2, High=3
    price_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df['Price_Value'] = df['Price_Range'].map(price_map)
    
    # กรองตามประเภทผิว (Target_Skin_Type)
    if skin_type != 'All':
        df = df[df['Target_Skin_Type'].isin([skin_type, 'All'])]

    # กรองตามหมวดหมู่ (Category)
    if category != 'All':
        df = df[df['Category'] == category]
        
    # กรองตามช่วงราคา (Price_Range)
    if max_price == 'Low':
        df = df[df['Price_Value'] <= 1]
    elif max_price == 'Medium':
        df = df[df['Price_Value'] <= 2]
    # ถ้าเลือก High จะรวมทุกช่วงราคา (Price_Value <= 3)

    # คืนค่าผลิตภัณฑ์ที่ดีที่สุด 5-10 รายการ
    return df.head(10).drop(columns=['Price_Value']).reset_index(drop=True)


# ----------------------------------------------------------------------
# 3. ฟังก์ชันจำลองการแนะนำเฉดสีรองพื้น
# ----------------------------------------------------------------------

def recommend_foundation(undertone, depth_scale, skin_type):
    """จำลองการจับคู่เฉดสีรองพื้นตามข้อมูลสีผิว/ความต้องการ"""
    if SHADE_DB.empty:
        return None

    # 1. กรองตาม Undertone (จำเป็น)
    filtered_df = SHADE_DB[SHADE_DB['Undertone'] == undertone]

    if filtered_df.empty:
        # ถ้าไม่มีโทนสีที่ตรง ให้ลองใช้ Neutral แทน
        filtered_df = SHADE_DB[SHADE_DB['Undertone'] == 'Neutral']
        if filtered_df.empty:
            return None # ไม่สามารถแนะนำได้

    # 2. จับคู่ Depth_Scale (ค้นหาสีที่ใกล้เคียง)
    # หาค่าเฉดสีที่ใกล้เคียงกับ Depth_Scale ที่ผู้ใช้ระบุที่สุด
    # เราใช้ numpy.abs เพื่อหาค่าสัมบูรณ์ของผลต่าง
    filtered_df['Depth_Diff'] = np.abs(filtered_df['Depth_Scale'] - depth_scale)
    filtered_df = filtered_df.sort_values(by='Depth_Diff').head(10) # เลือกมา 10 อันดับแรกที่ใกล้เคียง

    # 3. กรองตาม Skin Type (ถ้ามี)
    if skin_type != 'All':
        filtered_df = filtered_df[filtered_df['Best_For_Type'].isin([skin_type, 'All'])]

    # คืนค่าเฉดสีที่ดีที่สุด 3 อันดับแรก
    return filtered_df.head(3).drop(columns=['Depth_Diff']).reset_index(drop=True)


# ----------------------------------------------------------------------
# 4. โครงสร้าง Streamlit หลัก (main)
# ----------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AI Skincare & Makeup Advisor",
        layout="wide"
    )
    st.title("✨ AI Skincare & Makeup Advisor (สำหรับผิวคนไทย)")
    st.markdown("---")


    # ==================================================================
    # 4.1 ฟังก์ชันแนะนำผลิตภัณฑ์บำรุงผิว
    # ==================================================================
    st.header("🧴 1. ระบบแนะนำผลิตภัณฑ์บำรุงผิว")

    if PRODUCT_DB.empty:
        st.error("โปรดอัปโหลดไฟล์ 'products.csv' ที่มีข้อมูลผลิตภัณฑ์ลงใน GitHub")
    else:
        st.markdown("กรุณาระบุความต้องการและปัญหาผิวของคุณ:")
        
        # UI Input สำหรับ Skincare
        with st.container():
            col_skin, col_price, col_cat = st.columns(3)

            skin_type = col_skin.selectbox(
                "ประเภทผิวหลัก/ปัญหาผิว:",
                options=['All', 'Oily', 'Dry', 'Sensitive', 'Acne-Prone', 'Hyperpigmentation'],
                index=0
            )

            max_price = col_price.selectbox(
                "ช่วงราคาสูงสุดที่ต้องการ:",
                options=['Low', 'Medium', 'High']
            )
            
            category = col_cat.selectbox(
                "ประเภทผลิตภัณฑ์ที่ต้องการ:",
                options=['All', 'Cleanser', 'Toner', 'Essence', 'Serum', 'Moisturizer', 'Sunscreen', 'Treatment', 'Mask', 'Mist'],
            )

            if st.button("🔎 ค้นหาผลิตภัณฑ์ที่เหมาะสม", use_container_width=True):
                recommendations = recommend_skincare(skin_type, max_price, category)
                
                if not recommendations.empty:
                    st.subheader(f"✅ ผลิตภัณฑ์แนะนำสำหรับผิว '{skin_type}' (หมวดหมู่: {category}, ราคา: {max_price} และต่ำกว่า)")
                    
                    # แสดงผลในรูปแบบ DataFrame
                    st.dataframe(recommendations)
                    
                else:
                    st.warning("⚠️ ขออภัย ไม่พบผลิตภัณฑ์ที่ตรงกับเงื่อนไขทั้งหมด กรุณาลองปรับการกรอง")

    st.markdown("---")
    
    # ==================================================================
    # 4.2 ฟังก์ชันแนะนำเฉดสีรองพื้น (จำลอง)
    # ==================================================================
    st.header("🎨 2. ระบบแนะนำเฉดสีรองพื้น (Foundation)")
    
    if SHADE_DB.empty:
        st.warning("⚠️ ไม่สามารถรันฟังก์ชันแนะนำรองพื้นได้เนื่องจากไม่พบไฟล์ 'foundation_shades.csv'")
    else:
        st.markdown("กรุณาระบุข้อมูลสีผิวเพื่อจำลองการจับคู่เฉดสี:")
        
        col_tone, col_depth, col_type_f = st.columns(3)

        # 1. Input: Undertone (แทนการวิเคราะห์ภาพ)
        user_undertone = col_tone.selectbox(
            "โทนสีผิว (Undertone) ของคุณ:",
            options=['Warm', 'Cool', 'Neutral'],
            help="โทนสีผิวใต้ผิวหนัง: เหลือง/เขียว=Warm, ชมพู/แดง=Cool, กลาง=Neutral"
        )

        # 2. Input: Depth Scale (แทนการวิเคราะห์ภาพ)
        user_depth = col_depth.slider(
            "ความเข้มของสีผิว (1=อ่อนมาก, 10=เข้มมาก):",
            min_value=1.0, max_value=10.0, value=4.0, step=0.5
        )

        # 3. Input: Skin Type (สำหรับเนื้อรองพื้น)
        user_skin_type_f = col_type_f.selectbox(
            "ประเภทผิวสำหรับรองพื้น:",
            options=['All', 'Oily', 'Dry', 'Sensitive'],
            index=0
        )

        if st.button("🔍 แนะนำเฉดสีรองพื้น", type="primary", use_container_width=True):
            st.subheader(f"✅ ผลลัพธ์: เฉดสีที่แนะนำสำหรับโทน {user_undertone} & ระดับผิว {user_depth}")

            foundation_recommendations = recommend_foundation(
                user_undertone, user_depth, user_skin_type_f
            )

            if foundation_recommendations is not None and not foundation_recommendations.empty:
                st.dataframe(
                    foundation_recommendations[['Shade_Name', 'Brand', 'Coverage', 'Best_For_Type', 'Konvy_Link']],
                    use_container_width=True
                )
                st.info("คำแนะนำเหล่านี้เป็นผลจากการจับคู่ค่าความเข้มและโทนสีผิวที่ป้อน")
            else:
                st.warning("⚠️ ขออภัย ไม่พบเฉดสีที่ใกล้เคียงในฐานข้อมูล โปรดลองปรับค่าความเข้มของสีผิว")


# ตรวจสอบว่าโค้ดหลักกำลังถูกรันหรือไม่
if __name__ == '__main__':
    main()
