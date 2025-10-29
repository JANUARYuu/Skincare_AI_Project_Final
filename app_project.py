import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io

# พยายาม Import OpenCV (อาจล้มเหลวบน Cloud)
try:
    import cv2
    OPENCV_LOADED = True
except ImportError:
    # หาก import ไม่สำเร็จ จะใช้การจำลอง Input แทน
    OPENCV_LOADED = False

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
    st.warning("⚠️ Warning: ไม่พบไฟล์ 'foundation_shades.csv' ฟังก์ชันรองพื้นอาจมีปัญหา")
    SHADE_DB = pd.DataFrame() 

# โหลด Haar Cascade อย่างปลอดภัย
FACE_CASCADE = None
EYE_CASCADE = None
if OPENCV_LOADED:
    try:
        FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        EYE_CASCADE = cv2.CascadeClassifier('haarcascade_eye.xml')
        if FACE_CASCADE.empty() or EYE_CASCADE.empty():
            st.warning("⚠️ Warning: ไม่พบไฟล์ Haar Cascade (.xml) จะใช้การจำลอง Input แทน.")
            FACE_CASCADE = None
    except Exception:
        FACE_CASCADE = None


# ----------------------------------------------------------------------
# 2. ฟังก์ชันวิเคราะห์ใบหน้าและเฉดสีผิว (Robust Analysis)
# ----------------------------------------------------------------------

def analyze_skin_from_image(image):
    """
    ตรวจจับใบหน้า, ดวงตา, และจำลองการวิเคราะห์เฉดสีผิว/สภาพผิว
    คืนค่า: (ภาพที่มีกรอบ, Undertone, Depth, Skin_Concern)
    """
    # ค่าเริ่มต้น (Default values)
    detected_img = image
    undertone = None
    depth = None
    skin_concern = None
    
    if FACE_CASCADE is None:
        # หาก OpenCV หรือ Haar Cascade ไม่พร้อม จะแจ้งเตือนและส่งค่า None เพื่อเปลี่ยนไปใช้ Manual Input
        return detected_img, None, None, None 
        
    # แปลงภาพ PIL เป็น NumPy Array และแปลงเป็น Grayscale สำหรับ OpenCV
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return image, None, None, None # ไม่พบใบหน้า
    
    # เลือกใบหน้าแรกที่พบและวาดกรอบ
    x, y, w, h = faces[0]
    cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # **[!!! ส่วนจำลองการวิเคราะห์เฉดสีผิว !!!]**
    # ใช้ความกว้างของใบหน้าเป็นตัวกระตุ้นการจำลองผลลัพธ์
    if w > 300: 
        undertone = np.random.choice(["Warm", "Neutral"])
        depth = np.random.uniform(5.5, 8.0)
        skin_concern = np.random.choice(['Oily', 'Hyperpigmentation', 'Acne-Prone'])
    else:
        undertone = np.random.choice(["Neutral", "Cool"])
        depth = np.random.uniform(2.0, 5.0)
        skin_concern = np.random.choice(['Dry', 'Sensitive', 'Acne-Prone'])
        
    detected_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

    return detected_img, undertone, depth, skin_concern


# ----------------------------------------------------------------------
# 3. ฟังก์ชันแนะนำผลิตภัณฑ์บำรุงผิว (Skincare)
# ----------------------------------------------------------------------

def recommend_skincare(skin_type, max_price, category):
    # ... (โค้ดเดิม) ...
    if PRODUCT_DB.empty:
        return pd.DataFrame()
    
    df = PRODUCT_DB.copy()
    price_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df['Price_Value'] = df['Price_Range'].map(price_map)
    
    if skin_type != 'All':
        # กรองให้ตรงกับปัญหาผิวหรือสินค้าสำหรับผิว 'All'
        df = df[df['Target_Skin_Type'].isin([skin_type, 'All'])]

    if category != 'All':
        df = df[df['Category'] == category]
        
    if max_price == 'Low':
        df = df[df['Price_Value'] <= 1]
    elif max_price == 'Medium':
        df = df[df['Price_Value'] <= 2]

    return df.head(10).drop(columns=['Price_Value']).reset_index(drop=True)


# ----------------------------------------------------------------------
# 4. ฟังก์ชันแนะนำเฉดสีรองพื้น (Foundation)
# ----------------------------------------------------------------------

def recommend_foundation(undertone, depth_scale, skin_type):
    # ... (โค้ดเดิม) ...
    if SHADE_DB.empty:
        return pd.DataFrame()

    filtered_df = SHADE_DB[SHADE_DB['Undertone'] == undertone]
    if filtered_df.empty:
        filtered_df = SHADE_DB[SHADE_DB['Undertone'] == 'Neutral']
        if filtered_df.empty:
            return pd.DataFrame() 

    filtered_df['Depth_Diff'] = np.abs(filtered_df['Depth_Scale'] - depth_scale)
    filtered_df = filtered_df.sort_values(by='Depth_Diff').head(10)

    if skin_type != 'All':
        filtered_df = filtered_df[filtered_df['Best_For_Type'].isin([skin_type, 'All'])]

    return filtered_df.head(3).drop(columns=['Depth_Diff']).reset_index(drop=True)


# ----------------------------------------------------------------------
# 5. โครงสร้าง Streamlit หลัก (main)
# ----------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AI Skin & Makeup Analyzer",
        layout="wide"
    )
    st.title("✨ AI Skin & Makeup Analyzer: วิเคราะห์ผิวผ่านรูปภาพ")
    st.markdown("ระบบนี้สามารถ **วิเคราะห์ผิวจากรูปภาพ (ถ้าโค้ดรองรับ)** หรือ **ใช้ Input ด้วยตนเอง** เพื่อแนะนำผลิตภัณฑ์ที่เหมาะสม")
    st.markdown("---")

    # ==================================================================
    # 5.1 UI: การอัปโหลด/ถ่ายภาพ & Input ด้วยตนเอง (เพื่อความเสถียร)
    # ==================================================================
    
    st.subheader("📸 1. วิเคราะห์ผิว / ระบุข้อมูลด้วยตนเอง")
    
    if FACE_CASCADE is None:
        st.warning("⚠️ **สถานะ:** ฟังก์ชันวิเคราะห์จากรูปภาพไม่พร้อม (ขาดไฟล์ Haar Cascade หรือ OpenCV) กรุณาระบุข้อมูลด้วยตนเองด้านล่าง:")
        use_image_analysis = False
    else:
        st.info("✅ **สถานะ:** ฟังก์ชันวิเคราะห์จากรูปภาพพร้อมใช้งาน! หรือเลือก 'Input ด้วยตนเอง' หากภาพไม่ชัดเจน")
        use_image_analysis = st.checkbox("ใช้ Input จากรูปภาพ", value=False)

    
    undertone_result = None
    depth_result = None
    skin_concern_result = None
    
    if use_image_analysis:
        col_upload, col_camera = st.columns(2)
        with col_upload:
            input_image = st.file_uploader("อัปโหลดรูปภาพ", type=["png", "jpg", "jpeg"])
        with col_camera:
            input_image = st.camera_input("ถ่ายรูปด้วยกล้อง") if not input_image else input_image

        if input_image is not None:
            image = Image.open(input_image)
            with st.spinner("กำลังวิเคราะห์ใบหน้าและสีผิว..."):
                processed_img, undertone_result, depth_result, skin_concern_result = analyze_skin_from_image(image)
            
            # แสดงภาพที่ตรวจจับ
            st.image(processed_img, caption="ผลการตรวจจับ (ถ้ามี)", use_column_width=True)

    # -----------------------------------------------------------------
    # กำหนดค่า Input สุดท้าย
    # -----------------------------------------------------------------
    if undertone_result is None: # ใช้ Manual Input หากวิเคราะห์ภาพไม่ได้
        st.markdown("---")
        st.subheader("💡 ระบุค่าผิวด้วยตนเอง (Manual Input)")
        col_tone, col_depth, col_type = st.columns(3)

        undertone_result = col_tone.selectbox(
            "โทนสีผิว (Undertone):",
            options=['Warm', 'Cool', 'Neutral'],
            index=2 # Default to Neutral
        )

        depth_result = col_depth.slider(
            "ความเข้มของสีผิว (1=อ่อนมาก, 10=เข้มมาก):",
            min_value=1.0, max_value=10.0, value=4.5, step=0.5
        )

        skin_concern_result = col_type.selectbox(
            "ปัญหาผิวหลัก:",
            options=['Oily', 'Dry', 'Sensitive', 'Acne-Prone', 'Hyperpigmentation', 'All'],
            index=0
        )
        st.info("กรุณากดปุ่ม 'แสดงผลการแนะนำ' ด้านล่าง")

    
    st.markdown("---")
    
    # ==================================================================
    # 5.2 การแสดงผลการแนะนำ (ใช้ผลลัพธ์จาก Image/Manual Input)
    # ==================================================================
    
    if st.button("🚀 แสดงผลการแนะนำทั้งหมด", type="primary", use_container_width=True):
        if undertone_result is None or skin_concern_result is None:
            st.warning("โปรดป้อนข้อมูลหรืออัปโหลดรูปภาพใบหน้าก่อน")
            return
            
        # -----------------------------------------------------
        # ผลการวิเคราะห์ (สรุป)
        # -----------------------------------------------------
        st.subheader("📊 ผลการวิเคราะห์ผิว (สำหรับคำแนะนำ)")
        st.success(f"**โทนสีผิว (Undertone):** {undertone_result} | **ระดับความเข้ม (Depth):** {depth_result:.1f} | **ปัญหาผิวหลัก:** {skin_concern_result}")
        st.markdown("---")

        # -----------------------------------------------------
        # แนะนำรองพื้น (Foundation)
        # -----------------------------------------------------
        st.subheader("🎨 แนะนำเฉดสีรองพื้น (Foundation Recommendation)")

        foundation_recommendations = recommend_foundation(
            undertone_result, depth_result, skin_concern_result
        )

        if not foundation_recommendations.empty:
            st.markdown(f"**เฉดสีที่ใกล้เคียงที่สุด** สำหรับโทน **{undertone_result}** และผิวระดับ **{depth_result:.1f}**:")
            st.dataframe(
                foundation_recommendations[['Shade_Name', 'Brand', 'Coverage', 'Best_For_Type', 'Konvy_Link']],
                use_container_width=True
            )
        else:
            st.warning("ไม่พบเฉดสีที่ใกล้เคียงในฐานข้อมูล กรุณาตรวจสอบไฟล์ 'foundation_shades.csv'")

        st.markdown("---")

        # -----------------------------------------------------
        # แนะนำผลิตภัณฑ์บำรุงผิว (Skincare)
        # -----------------------------------------------------
        st.subheader("🧴 แนะนำผลิตภัณฑ์บำรุงผิว (Skincare Recommendation)")
        
        col_price, col_cat = st.columns(2)
        
        max_price_skincare = col_price.selectbox(
            "ช่วงราคาสูงสุดที่ต้องการสำหรับ Skincare:",
            options=['Low', 'Medium', 'High'],
            key="skincare_price"
        )
        
        category_skincare = col_cat.selectbox(
            "ประเภทผลิตภัณฑ์ที่ต้องการ:",
            options=['All', 'Cleanser', 'Toner', 'Essence', 'Serum', 'Moisturizer', 'Sunscreen', 'Treatment', 'Mask', 'Mist'],
            key="skincare_cat"
        )

        recommendations = recommend_skincare(skin_concern_result, max_price_skincare, category_skincare)
        
        if not recommendations.empty:
            st.markdown(f"**ผลิตภัณฑ์ที่แนะนำ** สำหรับผิวที่มีปัญหาหลัก **'{skin_concern_result}'**:")
            st.dataframe(recommendations)
        else:
            st.warning("ไม่พบผลิตภัณฑ์ที่ตรงกับเงื่อนไขทั้งหมด กรุณาตรวจสอบไฟล์ 'products.csv'")
            
    

if __name__ == '__main__':
    main()
