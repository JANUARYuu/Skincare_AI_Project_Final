import streamlit as st
import pandas as pd
import numpy as np
import cv2  # ต้องใช้ OpenCV
from PIL import Image
import io

# ----------------------------------------------------------------------
# 1. การโหลดฐานข้อมูลและโมเดล
# ----------------------------------------------------------------------

# โหลดฐานข้อมูล (สมมติว่าไฟล์อยู่ใน Root Directory)
try:
    PRODUCT_DB = pd.read_csv("products.csv")
except FileNotFoundError:
    st.error("❌ ERROR: ไม่พบไฟล์ 'products.csv'")
    PRODUCT_DB = pd.DataFrame() 

try:
    SHADE_DB = pd.read_csv("foundation_shades.csv")
except FileNotFoundError:
    st.warning("⚠️ Warning: ไม่พบไฟล์ 'foundation_shades.csv'")
    SHADE_DB = pd.DataFrame() 

# โหลด Haar Cascade สำหรับตรวจจับใบหน้าและดวงตา (ต้องอัปโหลดไฟล์ไป GitHub)
try:
    FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    EYE_CASCADE = cv2.CascadeClassifier('haarcascade_eye.xml')
    if FACE_CASCADE.empty() or EYE_CASCADE.empty():
        st.error("❌ ERROR: ไม่สามารถโหลดโมเดล Haar Cascade (.xml) ได้ โปรดตรวจสอบไฟล์ใน GitHub.")
except Exception as e:
    st.error(f"❌ ERROR: ไม่สามารถโหลด OpenCV/Haar Cascade ได้ ({e})")
    FACE_CASCADE = None
    EYE_CASCADE = None


# ----------------------------------------------------------------------
# 2. ฟังก์ชันวิเคราะห์ใบหน้าและเฉดสีผิว
# ----------------------------------------------------------------------

def detect_and_analyze_face(image):
    """
    ตรวจจับใบหน้า, ดวงตา, และจำลองการวิเคราะห์เฉดสีผิว
    คืนค่า: (ภาพที่มีกรอบ, Undertone, Depth)
    """
    if FACE_CASCADE is None or EYE_CASCADE is None:
        return image, None, None

    # แปลงภาพ PIL เป็น NumPy Array และแปลงเป็น Grayscale สำหรับ OpenCV
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
    
    # กำหนดค่าสีผิวจำลองจากผลการวิเคราะห์
    undertone = "Neutral"
    depth = 4.5
    
    if len(faces) == 0:
        return image, None, None # ไม่พบใบหน้า
    
    # เลือกใบหน้าแรกที่พบ
    for (x, y, w, h) in faces:
        # วาดกรอบสี่เหลี่ยมรอบใบหน้า
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Crop บริเวณใบหน้าเพื่อวิเคราะห์เฉดสี (สมมติว่าวิเคราะห์จากแก้ม/คอ)
        face_roi = img_array[y:y + h, x:x + w]
        
        # **[!!! ส่วนจำลองการวิเคราะห์เฉดสีผิวจริง !!!]**
        # ในการใช้งานจริง, โค้ดตรงนี้จะคำนวณค่าสี RGB/LAB ในบริเวณที่กำหนด
        # เพื่อหา Undertone (Warm/Cool/Neutral) และ Depth (ค่าความสว่าง)
        
        # ***โค้ดจำลอง:***
        # จำลองการตั้งค่าตามความกว้างของใบหน้า: กว้างมาก = ผิวเข้ม (Depth สูง)
        if w > 300:
            undertone = "Warm"
            depth = np.random.uniform(6.0, 8.0)
        else:
            undertone = "Neutral"
            depth = np.random.uniform(3.0, 5.0)
            
        # ตรวจจับดวงตาใน ROI ของใบหน้า
        roi_gray = gray[y:y + h, x:x + w]
        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img_array, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
            
        # ใช้แค่ใบหน้าแรก
        break

    return Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)), undertone, depth


# ----------------------------------------------------------------------
# 3. ฟังก์ชันแนะนำเฉดสีรองพื้น (ใช้จากโค้ดเดิม)
# ----------------------------------------------------------------------

def recommend_foundation(undertone, depth_scale, skin_type):
    """จำลองการจับคู่เฉดสีรองพื้นตามข้อมูลสีผิว/ความต้องการ"""
    if SHADE_DB.empty:
        return None

    filtered_df = SHADE_DB[SHADE_DB['Undertone'] == undertone]
    if filtered_df.empty:
        filtered_df = SHADE_DB[SHADE_DB['Undertone'] == 'Neutral']
        if filtered_df.empty:
            return None 

    # จับคู่ Depth_Scale
    filtered_df['Depth_Diff'] = np.abs(filtered_df['Depth_Scale'] - depth_scale)
    filtered_df = filtered_df.sort_values(by='Depth_Diff').head(10)

    # กรองตาม Skin Type
    if skin_type != 'All':
        filtered_df = filtered_df[filtered_df['Best_For_Type'].isin([skin_type, 'All'])]

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
    # 4.1 ฟังก์ชันแนะนำผลิตภัณฑ์บำรุงผิว (Skincare)
    # ==================================================================
    st.header("🧴 1. ระบบแนะนำผลิตภัณฑ์บำรุงผิว")
    # ... (นำโค้ดส่วน Skincare เดิมมาวางที่นี่) ...
    # UI Input สำหรับ Skincare
    if PRODUCT_DB.empty:
        st.error("โปรดอัปโหลดไฟล์ 'products.csv' ที่มีข้อมูลผลิตภัณฑ์ลงใน GitHub")
    else:
        st.markdown("กรุณาระบุความต้องการและปัญหาผิวของคุณ:")
        
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
                    st.dataframe(recommendations)
                else:
                    st.warning("⚠️ ขออภัย ไม่พบผลิตภัณฑ์ที่ตรงกับเงื่อนไขทั้งหมด กรุณาลองปรับการกรอง")


    st.markdown("---")


    # ==================================================================
    # 4.2 ฟังก์ชันแนะนำเฉดสีรองพื้นผ่านรูปภาพ (Foundation & Face Analysis)
    # ==================================================================
    st.header("🎨 2. ระบบวิเคราะห์ใบหน้าและแนะนำเฉดสีรองพื้น")
    
    if SHADE_DB.empty or FACE_CASCADE is None:
        st.warning("⚠️ ไม่สามารถรันฟังก์ชันนี้ได้: ขาดไฟล์ 'foundation_shades.csv' หรือไฟล์ Haar Cascade (.xml)")
    else:
        st.markdown("อัปโหลดรูปภาพใบหน้าของคุณเพื่อวิเคราะห์โทนสีผิวและแนะนำรองพื้น:")
        
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            # แปลงไฟล์ที่อัปโหลดเป็นภาพ PIL
            image = Image.open(uploaded_file)
            
            # รันฟังก์ชันวิเคราะห์
            processed_img, undertone, depth = detect_and_analyze_face(image)

            col_img, col_result = st.columns(2)
            
            with col_img:
                st.image(processed_img, caption="ผลการตรวจจับใบหน้าและดวงตา", use_column_width=True)

            with col_result:
                if undertone is None:
                    st.error("ไม่พบใบหน้าในภาพ กรุณาลองใช้ภาพที่ชัดเจนขึ้น")
                else:
                    st.subheader("📊 ผลการวิเคราะห์เบื้องต้น (จำลอง)")
                    st.info(f"**โทนสีผิว (Undertone):** {undertone}")
                    st.info(f"**ระดับความเข้ม (Depth):** {depth:.1f}")

                    # --- นำผลลัพธ์มาใช้แนะนำรองพื้น ---
                    st.subheader("✨ เฉดสีรองพื้นที่แนะนำ")
                    
                    # ถามประเภทผิว (สำหรับเนื้อรองพื้น)
                    user_skin_type_f = st.selectbox(
                        "ประเภทผิวสำหรับรองพื้น:",
                        options=['All', 'Oily', 'Dry', 'Sensitive'],
                        index=0
                    )
                    
                    foundation_recommendations = recommend_foundation(
                        undertone, depth, user_skin_type_f
                    )

                    if foundation_recommendations is not None and not foundation_recommendations.empty:
                        st.dataframe(
                            foundation_recommendations[['Shade_Name', 'Brand', 'Coverage', 'Best_For_Type', 'Konvy_Link']],
                            use_container_width=True
                        )
                    else:
                        st.warning("ไม่พบเฉดสีที่ใกล้เคียงในฐานข้อมูล")


if __name__ == '__main__':
    main()
