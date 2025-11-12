import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import io
import random
from PIL import Image
from collections import defaultdict

# ----------------------------------------------------------------------
# 1. การโหลดข้อมูลและตั้งค่า
# ----------------------------------------------------------------------

# ตั้งค่าหน้าเพจ
st.set_page_config(layout="wide", page_title="JVP Face Analyzer")

# ฟังก์ชันโหลดข้อมูล (แก้ไข NameError: เปลี่ยนชื่อจาก load_db_product_data เป็น load_db)
@st.cache_data
def load_db(file_path):
    """โหลดไฟล์ CSV และแปลงคอลัมน์ที่จำเป็น"""
    try:
        db = pd.read_csv(file_path, na_values=['N/A', '', ' '])
        
        # แปลง Depth_Scale เป็นตัวเลข (สำหรับ foundation)
        if 'Depth_Scale' in db.columns:
            db['Depth_Scale'] = pd.to_numeric(db['Depth_Scale'], errors='coerce') 
        
        # เติมค่าว่างใน Key_Ingredient/Key_Feature ด้วย 'ไม่มี' (สำหรับ Skincare/Makeup)
        if 'Key_Ingredient' in db.columns:
            db['Key_Ingredient'] = db['Key_Ingredient'].astype(str).fillna('ไม่มี')
        if 'Key_Feature' in db.columns:
            db['Key_Feature'] = db['Key_Feature'].astype(str).fillna('ไม่มี') # แก้ไข Key_feature เป็น Key_Feature
            
        if db.empty:
            st.warning(f"ไฟล์ '{file_path}' ดูเหมือนจะว่างเปล่า")
            return pd.DataFrame()
            
        return db

    except Exception as e:
        st.error(f"มีข้อผิดพลาดในการโหลดไฟล์ '{file_path}': {e}")
        return pd.DataFrame()

# โหลดฐานข้อมูล
PRODUCT_DB = load_db('products.csv')
SHADE_DB = load_db('foundation_shades.csv')
TONE_DB = load_db('skin_tones.csv')
MAKEUP_DB = load_db('makeup_products.csv')


# ----------------------------------------------------------------------
# 2. การตั้งค่า DNN (Deep Neural Network) และค่าคงที่
# ----------------------------------------------------------------------

# DNN (Deep Learning Model) หาใบหน้า (SSD)
PROTOTXT = 'deploy.prototxt'
CAFFEMODEL = 'res10_300x300_ssd_iter_140000.caffemodel'
CONFIDENCE_THRESHOLD = 0.7 
DNN_FACE_DETECTOR = None

if not os.path.exists(PROTOTXT) or not os.path.exists(CAFFEMODEL):
    st.error("❗ ไฟล์โมเดล DNN ไม่ครบ: กรุณาอัปโหลด deploy.prototxt และ res10_300x300_ssd_iter_140000.caffemodel")
else:
    try:
        DNN_FACE_DETECTOR = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
    except Exception as e:
        st.error(f"มีข้อผิดพลาดในการโหลดโมเดล DNN: {e}")
        DNN_FACE_DETECTOR = None


# ----------------------------------------------------------------------
# 3. ฟังก์ชันวิเคราะห์ภาพ (Image Analysis)
# ----------------------------------------------------------------------

def analyze_and_crop_face(image_file, detector):
    """วิเคราะห์ภาพ หาใบหน้า และ Crop"""
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob)
    detections = detector.forward()

    max_confidence = 0
    best_bbox = None

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD:
            if confidence > max_confidence:
                max_confidence = confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                best_bbox = box.astype("int")

    if best_bbox is not None:
        (startX, startY, endX, endY) = best_bbox
        
        padding = 30
        startX = max(0, startX - padding)
        startY = max(0, startY - padding)
        endX = min(w, endX + padding)
        endY = min(h, endY + padding)

        cropped_face = image[startY:endY, startX:endX]
        
        cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        return cropped_face_rgb
    
    return None

def predict_skin_condition(cropped_face_rgb):
    """
    ฟังก์ชันทำนายโทนสีผิว, ประเภทผิว และคะแนนสิว (Acne Score) 
    โดยจำลองคะแนนสิวตามค่าสีเฉลี่ย
    """
    
    avg_color = np.mean(cropped_face_rgb, axis=(0, 1))
    
    # 1. การทำนายโทนสีผิว (จำลอง)
    if avg_color[0] > 180 and avg_color[1] > 160 and avg_color[2] > 140:
        tone_group = random.choice(['Fair', 'Light'])
        undertone = random.choice(['Cool-Pink', 'Neutral'])
        acne_score = random.choice([1, 1, 2, 2, 3]) # ผิวสีอ่อน มักมีสิวน้อย (1-3)
    elif avg_color[0] < 120 and avg_color[1] < 100 and avg_color[2] < 90:
        tone_group = random.choice(['Deep', 'Dark'])
        undertone = random.choice(['Warm-Olive', 'Warm'])
        acne_score = random.choice([2, 3, 3, 4, 5]) # ผิวสีเข้ม อาจมีปัญหาเม็ดสี/สิวมากกว่า (2-5)
    else:
        tone_group = random.choice(['Medium', 'Tan'])
        undertone = random.choice(['Neutral', 'Warm'])
        acne_score = random.choice([2, 2, 3, 3, 4]) # ผิวสีกลางๆ (2-4)
        
    # 2. การทำนายประเภทผิว (จำลอง)
    skin_type = random.choice(['Oily', 'Combination', 'Normal', 'Dry', 'Sensitive'])

    return tone_group, undertone, skin_type, acne_score 


# ----------------------------------------------------------------------
# 4. ฟังก์ชันแนะนำผลิตภัณฑ์ (Recommendation Logic)
# ----------------------------------------------------------------------

def get_skincare_recommendation(user_skin_type, user_acne_score, product_db):
    """Logic การแนะนำ Skincare ตามคะแนนสิว (1-5) ที่ถูก AI จำลองขึ้นมา"""
    
    # Logic ที่ทำให้คนหน้าใส (1) กับคนเป็นสิวเยอะ (4-5) ได้ผลลัพธ์ต่างกัน
    if user_acne_score <= 1:
        target_ingredients = ['Ceramide', 'Hyaluronic Acid', 'Vitamin C', 'SPF50+']
        recommendation_text = "**ผิวสวยใส** สกินแคร์ควรเน้นการบำรุง เติมความชุ่มชื้น และป้องกันผิวจากแสงแดดเป็นหลัก (Sunscreen/Moisturizer)"
    
    elif user_acne_score == 2:
        target_ingredients = ['Salicylic Acid|BHA', 'Centella Asiatica', 'Lightweight', 'Gel']
        recommendation_text = "**สิวเล็กน้อย** ควรใช้ Cleanser/Toner ที่มี BHA อ่อนโยน และเพิ่ม Spot Treatment หากจำเป็น (Treatment/Cleanser)"
    
    elif user_acne_score == 3:
        target_ingredients = ['Benzoyl Peroxide', 'Salicylic Acid|BHA', 'Retinol|Retinal', 'Oil Control']
        recommendation_text = "**สิวปานกลาง** ควรเน้น Treatment ที่มีสารรักษาสิวเข้มข้น และมอยส์เจอไรเซอร์สูตรอ่อนโยน เพื่อไม่ให้อุดตัน (Treatment/Moisturizer)"

    elif user_acne_score >= 4:
        target_ingredients = ['Retinol|Retinal', 'Benzoyl Peroxide', 'Soothes', 'Emulsion']
        recommendation_text = "**สิวอักเสบรุนแรง** ควรปรึกษาแพทย์ผิวหนัง ควบคู่กับการใช้ผลิตภัณฑ์ที่เน้นการรักษาสิว และบรรเทาอาการอักเสบ (Treatment/Emulsion)"
        
    else:
        target_ingredients = ['Hyaluronic Acid', 'Glycerin']
        recommendation_text = "ไม่สามารถประเมินสิวได้ แต่เน้นความชุ่มชื้นไว้ก่อน"
        
    # กรองผลิตภัณฑ์ตามส่วนผสมหลัก
    filtered_products = product_db[
        product_db['Key_Ingredient'].str.contains('|'.join(target_ingredients), case=False, na=False)
    ]
    
    # ปรับการกรองตามประเภทผิว (เดิม)
    if user_skin_type in ['Oily', 'Combination']:
        filtered_products = filtered_products[~filtered_products['Product_Name'].str.contains('Oil|Balm', case=False)]
    elif user_skin_type in ['Dry', 'Sensitive']:
        filtered_products = filtered_products[filtered_products['Category'].isin(['Moisturizer', 'Cleanser', 'Sunscreen'])]
        
    return filtered_products.head(5), recommendation_text


def get_foundation_recommendation(user_undertone, shade_db):
    """แนะนำ Foundation ตาม Undertone"""
    filtered_shades = shade_db[shade_db['Undertone'] == user_undertone]
    return filtered_shades.head(5)

def get_makeup_recommendation(user_undertone, makeup_db):
    """แนะนำ Makeup ตาม Undertone"""
    filtered_makeup = makeup_db[makeup_db['Tone_Type'].str.contains(user_undertone.split('-')[0], case=False, na=False)]
    return filtered_makeup.head(5)


# ----------------------------------------------------------------------
# 5. UI และการแสดงผล (Streamlit UI)
# ----------------------------------------------------------------------

st.title("JVP Face Analyzer 🧖‍♀️💄")
st.markdown("ระบบวิเคราะห์โทนสีผิวและลักษณะใบหน้า เพื่อแนะนำผลิตภัณฑ์ที่เหมาะสม **(การประเมินสิวเป็นแบบจำลองอัตโนมัติจากภาพ)**")

# Upload File Section
st.subheader("อัปโหลดรูปภาพใบหน้าของคุณ 📸")
uploaded_file = st.file_uploader(
    "เคล็ดลับ: ใช้ภาพที่เห็นใบหน้าเต็มชัดเจนและอยู่ในแสงธรรมชาติ",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    if DNN_FACE_DETECTOR is not None:
        
        # 1. วิเคราะห์ภาพ
        uploaded_file.seek(0)
        cropped_face = analyze_and_crop_face(uploaded_file, DNN_FACE_DETECTOR)

        if cropped_face is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ ใบหน้าส่วนที่ Crop")
                st.image(cropped_face, caption="ใบหน้าที่ถูก Crop เพื่อวิเคราะห์", use_column_width=True)
                
            # 2. ทำนายโทนสีผิว, ประเภทผิว, และคะแนนสิว
            tone_group, undertone, skin_type, ai_acne_score = predict_skin_condition(cropped_face)

            with col2:
                st.subheader("✨ ผลการวิเคราะห์ผิว")
                st.markdown(f"**โทนสีผิวหลัก (Tone Group):** <span style='background-color:#ffe4b5; padding: 4px; border-radius: 5px;'>{tone_group}</span>", unsafe_allow_html=True)
                st.markdown(f"**อันเดอร์โทน (Undertone):** <span style='background-color:#add8e6; padding: 4px; border-radius: 5px;'>{undertone}</span>", unsafe_allow_html=True)
                st.markdown(f"**ประเภทผิว (Skin Type):** <span style='background-color:#90ee90; padding: 4px; border-radius: 5px;'>{skin_type}</span>", unsafe_allow_html=True)
                st.markdown(f"**คะแนนปัญหาสิว (AI ประเมิน):** <span style='background-color:#f08080; color:white; padding: 4px; border-radius: 5px;'>{ai_acne_score}</span>", unsafe_allow_html=True)


            st.markdown("---")
            
            # 3. การแนะนำผลิตภัณฑ์
            st.subheader("🛒 ผลิตภัณฑ์แนะนำสำหรับคุณ")
            
            # Skincare Recommendation (ใช้คะแนนที่ AI วิเคราะห์ได้)
            skincare_recs, skincare_text = get_skincare_recommendation(skin_type, ai_acne_score, PRODUCT_DB)
            st.markdown(f"#### 🧴 Skincare Recommendation: {skincare_text}")
            st.dataframe(skincare_recs[['Product_Name', 'Brand', 'Category', 'Key_Ingredient', 'Price_Range']], hide_index=True, use_container_width=True)
            
            st.markdown("---")
            
            # Makeup & Foundation Recommendation
            st.subheader("💄 Makeup Recommendation")
            
            col_fd, col_mk = st.columns(2)
            
            with col_fd:
                st.markdown("##### 👩‍🦰 Foundation/Concealer Shades (Undertone Match)")
                foundation_recs = get_foundation_recommendation(undertone, SHADE_DB)
                st.dataframe(foundation_recs[['Brand', 'Shade_Name', 'Coverage', 'Price_Range']], hide_index=True, use_container_width=True)
                
            with col_mk:
                st.markdown("##### 💋 Makeup Products (Blush, Lip, Contour)")
                makeup_recs = get_makeup_recommendation(undertone, MAKEUP_DB)
                st.dataframe(makeup_recs[['Product_Name', 'Brand', 'Category', 'Tone_Type', 'Price_Range']], hide_index=True, use_container_width=True)

        else:
            st.error("ไม่พบใบหน้าในภาพ: กรุณาลองภาพที่มีใบหน้าชัดเจน")
            
    else:
        st.warning("ระบบตรวจจับใบหน้า DNN ไม่พร้อมใช้งาน (ไฟล์โมเดลขาดหาย)")
        
else:
    st.info("กรุณาอัปโหลดรูปภาพใบหน้าเพื่อเริ่มต้นการวิเคราะห์")