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
        # กำหนด dtypes เพื่อป้องกันปัญหา 'float' และ 'str' ในคอลัมน์ตัวเลข
        if file_path == 'foundation_shades.csv' or file_path == 'skin_tones.csv':
            dtype_map = {'Depth_Scale': float}
        else:
            dtype_map = None

        db = pd.read_csv(file_path, dtype=dtype_map, na_values=['N/A', '', ' '])
        
        # สำหรับคอลัมน์ตัวเลข ให้เติมค่าว่างด้วย NaN และแปลงเป็นตัวเลขเพื่อคำนวณ
        if 'Depth_Scale' in db.columns:
            db['Depth_Scale'] = pd.to_numeric(db['Depth_Scale'], errors='coerce')
        
        # เตรียมข้อมูลสำหรับ Rule-Based Logic
        if 'Key_Ingredient' in db.columns:
             db['Key_Ingredient'] = db['Key_Ingredient'].astype(str).fillna('ไม่ระบุ') 
        if 'Price_Range' in db.columns:
            db['Price_Range'] = db['Price_Range'].fillna('ไม่ระบุ')
            
        return db
    except Exception as e:
        st.error(f"❗ ข้อผิดพลาดในการอ่านไฟล์ {file_path}: {e}")
        return pd.DataFrame()

# โหลดฐานข้อมูลทั้งหมด
PRODUCT_DB = load_db('products.csv')
SHADE_DB = load_db('foundation_shades.csv')
TONE_DB = load_db('skin_tones.csv')
MAKEUP_DB = load_db('makeup_products.csv')


# ----------------------------------------------------------------------
# 1. ฟังก์ชันจำลองการวิเคราะห์สภาพผิวและโทนสีผิว (Simulation)
# ----------------------------------------------------------------------

def analyze_skin(uploaded_file, tone_db):
    """ฟังก์ชันจำลองผลการวิเคราะห์ผิวหน้าตามชื่อไฟล์ที่อัปโหลด"""
    if uploaded_file is None or tone_db.empty:
        return None
        
    file_name = uploaded_file.name.lower()
    
    # 1. จำลองปัญหาผิวหลัก (Skin Concern)
    if any(keyword in file_name for keyword in ["acne", "oil", "มัน", "สิว"]):
        skin_type = 'Oily'  
        acne_severity = 'Moderate'
        tone_options = tone_db[tone_db['Depth_Scale'].between(3.5, 6.5)]
    elif any(keyword in file_name for keyword in ["dry", "แห้ง", "sensitive", "แพ้"]):
        skin_type = 'Dry' 
        acne_severity = 'Low'
        tone_options = tone_db[tone_db['Depth_Scale'].between(1.0, 3.5)]
    else:
        skin_type = 'Combination' 
        acne_severity = 'Low'
        tone_options = tone_db[tone_db['Depth_Scale'].between(3.0, 5.0)]
        
    # 2. สุ่มผลลัพธ์โทนสีผิวจากฐานข้อมูลที่กรองแล้ว
    if not tone_options.empty:
        random_tone = tone_options.sample(n=1).iloc[0]
        # ตรวจสอบค่าก่อนนำมาใช้
        undertone = random_tone['Undertone'] if 'Undertone' in random_tone else 'Neutral'
        depth_scale = random_tone['Depth_Scale'] if 'Depth_Scale' in random_tone and not pd.isna(random_tone['Depth_Scale']) else 4.5
    else:
        undertone = 'Neutral'
        depth_scale = 4.5
        
    return {
        'Skin_Type': skin_type,  
        'Acne_Severity': acne_severity,
        'Undertone': undertone,
        'Depth_Scale': float(depth_scale) # รับรองว่าเป็น float
    }

# ----------------------------------------------------------------------
# 2.1 ฟังก์ชันแนะนำผลิตภัณฑ์บำรุงผิว (Skincare - Rule-Based Logic)
# ----------------------------------------------------------------------

def recommend_skincare(skin_analysis_results, db):
    """กำหนดกฎเกณฑ์การแนะนำผลิตภัณฑ์บำรุงผิว"""
    skin_type = skin_analysis_results['Skin_Type']
    acne_severity = skin_analysis_results['Acne_Severity']
    recommendations = {}

    # 1. Cleanser
    if skin_type in ['Oily', 'Combination'] and acne_severity != 'Low':
        reco = db[(db['Category'] == 'Cleanser') & (db['Key_Ingredient'].str.contains('Salicylic Acid|BHA', case=False))].head(1)
    elif skin_type == 'Dry':
        reco = db[(db['Category'] == 'Cleanser') & (db['Key_Ingredient'].str.contains('Ceramide|Glycerin', case=False))].head(1)
    else:
        reco = db[db['Category'] == 'Cleanser'].head(1)
    if not reco.empty:
        recommendations['Step 1: Cleanser (ทำความสะอาด)'] = reco

    # 2. Treatment
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
        
    # กรองข้อมูลให้เหลือเฉพาะแถวที่ Depth_Scale เป็นตัวเลข (ไม่เป็น NaN)
    filtered_df = db[db['Depth_Scale'].notna()]
    
    # 1. กรองตาม Undertone ที่ตรงกัน
    filtered_df = filtered_df[filtered_df['Undertone'] == undertone]
    
    # 2. หากไม่พบ ให้ใช้ Neutral แทน
    if filtered_df.empty:
        filtered_df = db[db['Undertone'] == 'Neutral']

    if filtered_df.empty:
        return pd.DataFrame() 

    # 3. คำนวณค่าความแตกต่าง (Depth_Scale ถูกรับรองว่าเป็น float แล้ว)
    filtered_df['Depth_Diff'] = np.abs(filtered_df['Depth_Scale'] - depth_scale)
    
    return filtered_df.sort_values(by='Depth_Diff').head(3).drop(columns=['Depth_Diff']).reset_index(drop=True)


# ----------------------------------------------------------------------
# 2.3 ฟังก์ชันแนะนำผลิตภัณฑ์แต่งหน้าอื่นๆ (Makeup)
# ----------------------------------------------------------------------

def recommend_makeup(undertone, db):
    """แนะนำผลิตภัณฑ์เมคอัพอื่นๆ ตามโทนผิว (Undertone)"""
    if db.empty:
        return pd.DataFrame()

    recommendations = {}

    # 1. แป้ง (Powder)
    reco_powder = db[db['Category'] == 'Powder'].head(1)
    if not reco_powder.empty:
        recommendations['Step 1: Powder (แป้ง)'] = reco_powder

    # 2. บลัชออน (Blush): จับคู่สีตามโทนผิว
    if undertone == 'Warm':
        color_keywords = 'Peach|Orange|Gold|Warm'
    elif undertone == 'Cool':
        color_keywords = 'Rose|Pink|Berry|Cool'
    else:
        color_keywords = 'Nude|Rose|Peach'
        
    reco_blush = db[(db['Category'] == 'Blush') & (db['Key_Feature'].str.contains(color_keywords, case=False, na=False))].head(1)
    if not reco_blush.empty:
        recommendations[f'Step 2: Blush ({undertone} Tone Match)'] = reco_blush

    # 3. ลิปสติก (Lip)
    reco_lip = db[(db['Category'] == 'Lip') & (db['Key_Feature'].str.contains(color_keywords, case=False, na=False))].head(1)
    if not reco_lip.empty:
        recommendations[f'Step 3: Lip Color ({undertone} Tone Match)'] = reco_lip

    # 4. Highlight/Contour
    reco_contour = db[(db['Category'] == 'Contour') & (db['Tone_Type'].isin(['Neutral', undertone]))].head(1)
    if not reco_contour.empty:
        recommendations['Step 4: Contour'] = reco_contour
        
    return recommendations


# ----------------------------------------------------------------------
# 3. Streamlit UI หลัก
# ----------------------------------------------------------------------

def main():
    # ตรวจสอบว่าไฟล์ฐานข้อมูลทั้งหมดถูกโหลดหรือไม่
    if PRODUCT_DB.empty or SHADE_DB.empty or TONE_DB.empty or MAKEUP_DB.empty:
        st.warning("❗ โปรดตรวจสอบว่าไฟล์ฐานข้อมูลทั้งหมด (products.csv, foundation_shades.csv, skin_tones.csv, makeup_products.csv) พร้อมใช้งานและมีข้อมูล")
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
                # แสดงภาพโดยไม่มีการประมวลผลใบหน้า (แก้ปัญหา Haar Cascade)
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
        st.header("🧴 2. ผลิตภัณฑ์บำรุงผิวที่แนะนำ (Skincare)")
        skincare_recommendations = recommend_skincare(results, PRODUCT_DB)

        if not skincare_recommendations:
            st.warning("ไม่พบผลิตภัณฑ์บำรุงผิวที่ตรงกับเงื่อนไขในฐานข้อมูล 'products.csv'.")
        else:
            for category, df_reco in skincare_recommendations.items():
                st.markdown(f"#### {category}")
                
                for _, row in df_reco.iterrows():
                    col_img_sk, col_info_sk = st.columns([1, 4]) 

                    with col_img_sk:
                        image_file = row.get('Image_File', 'default.png')
                        image_path = f"images/{image_file}"
                        if os.path.exists(image_path):
                            st.image(image_path, width=100)
                        else:
                            st.caption(f"No Image: {image_file}")
                            
                    with col_info_sk:
                        st.markdown(f"**{row['Product_Name']}** (แบรนด์: {row['Brand']})")
                        st.markdown(f"**จุดเด่น:** *{row['Key_Ingredient']}* | ราคา: {row['Price_Range']}")
                        
                    st.markdown("---")
        
        st.markdown("---")

        # 3. แสดงผลการแนะนำ Foundation
        st.header("🎨 3. เฉดสีรองพื้นที่แนะนำ (Foundation)")
        
        foundation_recommendations = recommend_foundation(
            results['Undertone'], results['Depth_Scale'], SHADE_DB
        )

        if not foundation_recommendations.empty:
            st.markdown(f"**เฉดสีที่ใกล้เคียงที่สุด** สำหรับโทน **{results['Undertone']}** และผิวระดับ **{results['Depth_Scale']:.1f}**:")
            
            for _, row in foundation_recommendations.iterrows():
                col_img_fd, col_info_fd = st.columns([1, 4]) 
                
                with col_img_fd:
                    image_file = row.get('Image_File', 'default.png')
                    image_path = f"images/{image_file}"
                    if os.path.exists(image_path):
                        st.image(image_path, width=100)
                    else:
                        st.caption(f"No Image: {image_file}")

                with col_info_fd:
                    # ตรวจสอบว่าคอลัมน์ Depth_Scale เป็นตัวเลขก่อนแสดงผล
                    depth_display = f"{row['Depth_Scale']:.1f}" if pd.notna(row['Depth_Scale']) else 'N/A'
                    st.markdown(f"**{row['Shade_Name']}** (แบรนด์: {row['Brand']})")
                    st.markdown(f"**ระดับ:** {row['Coverage']} | **โทน:** {row['Undertone']} | **Depth:** {depth_display}")

                st.markdown("---")
        else:
            st.warning("ไม่พบเฉดสีที่ใกล้เคียงในฐานข้อมูล 'foundation_shades.csv'.")
            
        st.markdown("---")
            
        # 4. แสดงผลการแนะนำ Makeup Products อื่นๆ
        st.header("💄 4. ผลิตภัณฑ์แต่งหน้าอื่นๆ ที่แนะนำ (Makeup)")

        if MAKEUP_DB.empty:
            st.warning("ไม่พบไฟล์ 'makeup_products.csv' กรุณาตรวจสอบ")
        else:
            makeup_recommendations = recommend_makeup(
                results['Undertone'], MAKEUP_DB
            )

            if not makeup_recommendations:
                st.warning("ไม่พบผลิตภัณฑ์เมคอัพที่ตรงกับเงื่อนไขในฐานข้อมูล")
            else:
                for category, df_reco in makeup_recommendations.items():
                    st.markdown(f"#### {category}")
                    
                    for _, row in df_reco.iterrows():
                        col_img_mk, col_info_mk = st.columns([1, 4]) 
                        
                        with col_img_mk:
                            image_file = row.get('Image_File', 'default.png')
                            image_path = f"images/{image_file}"
                            if os.path.exists(image_path):
                                st.image(image_path, width=100)
                            else:
                                st.caption(f"No Image: {image_file}")

                        with col_info_mk:
                            st.markdown(f"**{row['Product_Name']}** (แบรนด์: {row['Brand']})")
                            st.markdown(f"**จุดเด่น:** *{row.get('Key_Feature', 'ไม่ระบุ')}* | ราคา: {row['Price_Range']}")

                        st.markdown("---")


if __name__ == "__main__":
    main()
