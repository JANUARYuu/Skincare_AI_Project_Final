import streamlit as st
import pandas as pd
import numpy as np
import cv2  # ต้องติดตั้ง: pip install opencv-python
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
FACE_CASCADE = None
EYE_CASCADE = None
try:
    # ตรวจสอบว่าไฟล์โมเดล Haar Cascade อยู่ใน GitHub หรือไม่
    FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    EYE_CASCADE = cv2.CascadeClassifier('haarcascade_eye.xml')
    if FACE_CASCADE.empty() or EYE_CASCADE.empty():
        st.error("❌ ERROR: ไม่สามารถโหลดโมเดล Haar Cascade (.xml) ได้ โปรดตรวจสอบไฟล์ใน GitHub.")
except Exception as e:
    # หาก Streamlit Cloud รันไม่ได้เพราะขาดโมเดล เราจะยังรันโค้ดส่วนอื่นต่อได้
    pass


# ----------------------------------------------------------------------
# 2. ฟังก์ชันวิเคราะห์ใบหน้าและเฉดสีผิว (Simulated Analysis)
# ----------------------------------------------------------------------

def detect_and_analyze_face(image):
    """
    ตรวจจับใบหน้า, ดวงตา, และจำลองการวิเคราะห์เฉดสีผิว/สภาพผิว
    คืนค่า: (ภาพที่มีกรอบ, Undertone, Depth, Skin_Concern)
    """
    # ค่าเริ่มต้น (Default values)
    detected_img = image
    undertone = None
    depth = None
    skin_concern = "All"
    
    if FACE_CASCADE is None or EYE_CASCADE is None:
        # หากไม่มีโมเดล cascade, ใช้การจำลองแบบสุ่มเท่านั้น
        undertone = np.random.choice(["Warm", "Cool", "Neutral"])
        depth = np.random.uniform(2.5, 7.5)
        skin_concern = np.random.choice(['Oily', 'Dry', 'Sensitive', 'Acne-Prone', 'Hyperpigmentation'])
        return detected_img, undertone, depth, skin_concern
        
    # แปลงภาพ PIL เป็น NumPy Array และแปลงเป็น Grayscale สำหรับ OpenCV
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return image, None, None, None # ไม่พบใบหน้า
    
    # เลือกใบหน้าแรกที่พบ
    x, y, w, h = faces[0]
    
    # วาดกรอบสี่เหลี่ยมรอบใบหน้า
    cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # **[!!! ส่วนจำลองการวิเคราะห์เฉดสีผิวจริง !!!]**
    # ในการใช้งานจริง โค้ดตรงนี้จะวิเคราะห์ค่าสีจากบริเวณที่กำหนด (เช่น คอหรือแก้ม)
    
    # ***โค้ดจำลอง:***
    # 1. Depth & Undertone (ใช้ความกว้างของใบหน้าและค่าสุ่ม)
    if w > 300: # สมมติฐาน: ความกว้างของใบหน้าสัมพันธ์กับความเข้มของผิว
        undertone = np.random.choice(["Warm", "Neutral"])
        depth = np.random.uniform(5.5, 8.0)
    else:
        undertone = np.random.choice(["Neutral", "Cool"])
        depth = np.random.uniform(2.0, 5.0)
        
    # 2. Skin Concern (จำลองตามตำแหน่งใบหน้า)
    # สมมติฐาน: หากใบหน้าอยู่ตรงกลางมาก (x น้อย) = มีสิว (Acne-Prone)
    if x < 50:
        skin_concern = 'Acne-Prone'
    # สมมติฐาน: หากใบหน้ามีขนาดเล็ก (w, h น้อย) = Sensitive
    elif w < 200:
        skin_concern = 'Sensitive'
    else:
        skin_concern = 'Oily' # ค่าสุ่มถ้าไม่เข้าเงื่อนไข
        
    # แปลงภาพกลับ
    detected_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

    return detected_img, undertone, depth, skin_concern


# ----------------------------------------------------------------------
# 3. ฟังก์ชันแนะนำผลิตภัณฑ์บำรุงผิว (Skincare - ใช้โค้ดเดิม)
# ----------------------------------------------------------------------

def recommend_skincare(skin_type, max_price, category):
    """ฟังก์ชันแนะนำผลิตภัณฑ์บำรุงผิวตามประเภทผิว, ราคา และหมวดหมู่"""
    if PRODUCT_DB.empty:
        return pd.DataFrame()
    
    df = PRODUCT_DB.copy()
    price_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df['Price_Value'] = df['Price_Range'].map(price_map)
    
    if skin_type != 'All':
        df = df[df['Target_Skin_Type'].isin([skin_type, 'All'])]

    if category != 'All':
        df = df[df['Category'] == category]
        
    if max_price == 'Low':
        df = df[df['Price_Value'] <= 1]
    elif max_price == 'Medium':
        df = df[df['Price_Value'] <= 2]

    return df.head(10).drop(columns=['Price_Value']).reset_index(drop=True)


# ----------------------------------------------------------------------
# 4. ฟังก์ชันแนะนำเฉดสีรองพื้น (Foundation - ใช้โค้ดเดิม)
# ----------------------------------------------------------------------

def recommend_foundation(undertone, depth_scale, skin_type):
    """จับคู่เฉดสีรองพื้นตามข้อมูลสีผิว/ความต้องการ"""
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
    st.markdown("ระบบจะวิเคราะห์โทนสีผิว ปัญหาผิว และแนะนำผลิตภัณฑ์ที่เหมาะสม")
    st.markdown("---")

    # ==================================================================
    # UI: การอัปโหลด/ถ่ายภาพ
    # ==================================================================
    
    st.subheader("📸 1. อัปโหลด/ถ่ายรูปใบหน้าของคุณ")
    col_upload, col_camera = st.columns(2)
    
    with col_upload:
        uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["png", "jpg", "jpeg"])
