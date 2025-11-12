import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import io
from PIL import Image

# ----------------------------------------------------------------------
# 1. การโหลดข้อมูลและตั้งค่า
# ----------------------------------------------------------------------

# ตั้งค่าหน้าเพจ
st.set_page_config(layout="wide", page_title="AI Skincare & Makeup Advisor: Image-Only Face Analysis")

# ฟังก์ชันโหลดข้อมูล (แก้ไข NameError: เปลี่ยนชื่อจาก load_db_product_data เป็น load_db)
@st.cache_data
def load_db(file_path):
    """โหลดไฟล์ CSV และแปลงคอลัมน์ที่จำเป็น"""
    try:
        db = pd.read_csv(file_path, na_values=['N/A', '', ' '])
        
        # แปลง Depth_Scale เป็นตัวเลข (สำหรับ foundation)
        if 'Depth_Scale' in db.columns:
            # แก้ไขข้อผิดพลาดด้านล่างในภาพ 04.20.42.png 
            db['Depth_Scale'] = pd.to_numeric(db['Depth_Scale'], errors='coerce') 
        
        # เติมค่าว่างใน Key_Ingredient/Key_Feature ด้วย 'ไม่มี' (สำหรับ Skincare/Makeup)
        if 'Key_Ingredient' in db.columns:
            db['Key_Ingredient'] = db['Key_Ingredient'].astype(str).fillna('ไม่มี')
        if 'Key_Feature' in db.columns:
            db['Key_Feature'] = db['Key_Feature'].astype(str).fillna('ไม่มี')

        if db.empty:
            st.warning(f"ไฟล์ '{file_path}' ดูเหมือนจะว่างเปล่า")
            return pd.DataFrame()
            
        return db

    except Exception as e:
        st.error(f"มีข้อผิดพลาดในการโหลดไฟล์ '{file_path}': {e}")
        return pd.DataFrame()

# โหลดฐานข้อมูล (แก้ไข NameError: เปลี่ยนการเรียกฟังก์ชันเป็น load_db)
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
CONFIDENCE_THRESHOLD = 0.7 # 0.7 คือค่าความมั่นใจในการตรวจจับ (70%)
DNN_FACE_DETECTOR = None

# แก้ไข IndentationError: ตรวจสอบความมั่นใจในไฟล์ DNN ว่ามีอยู่จริงหรือไม่
# ข้อผิดพลาด IndentationError ในภาพ 03.27.16.png ถูกแก้ไขแล้วในบล็อกนี้
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
    # ... (ส่วนนี้ใช้ OpenCV และ DNN ในการประมวลผล)
    
    # แปลงไฟล์ภาพที่อัปโหลดเป็นฟอร์แมตที่ OpenCV อ่านได้
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    (h, w) = image.shape[:2]

    # สร้าง blob จากภาพและส่งเข้าโมเดล
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob)
    detections = detector.forward()

    max_confidence = 0
    best_bbox = None

    # วนลูปหาใบหน้าที่มั่นใจที่สุด
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD:
            if confidence > max_confidence:
                max_confidence = confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                best_bbox = box.astype("int")

    if best_bbox is not None:
        (startX, startY, endX, endY) = best_bbox
        
        # ขยายกรอบ Crop เล็กน้อยเพื่อไม่ให้ภาพถูกตัด
        padding = 30
        startX = max(0, startX - padding)
        startY = max(0, startY - padding)
        endX = min(w, endX + padding)
        endY = min(h, endY + padding)

        # Crop ใบหน้า
        cropped_face = image[startY:endY, startX:endX]
        
        # แปลงเป็น RGB (Streamlit แสดงผลเป็น RGB)
        cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        return cropped_face_rgb
    
    return None

def predict_skin_tone_and_type(cropped_face_rgb):
    """ฟังก์ชันจำลองการทำนายโทนสีผิวและประเภทผิว"""
    
    # 1. การทำนายโทนสีผิว (จำลอง)
    # ในโปรเจกต์จริง ส่วนนี้จะซับซ้อนกว่านี้
    avg_color = np.mean(cropped_face_rgb, axis=(0, 1))
    
    if avg_color[0] > 180 and avg_color[1] > 160 and avg_color[2] > 140:
        tone_group = random.choice(['Fair', 'Light'])
        undertone = random.choice(['Cool-Pink', 'Neutral'])
    elif avg_color[0] < 120 and avg_color[1] < 100 and avg_color[2] < 90:
        tone_group = random.choice(['Deep', 'Dark'])
        undertone = random.choice(['Warm-Olive', 'Warm'])
    else:
        tone_group = random.choice(['Medium', 'Tan'])
        undertone = random.choice(['Neutral', 'Warm'])
        
    # 2. การทำนายประเภทผิว (จำลอง)
    skin_type = random.choice(['Oily', 'Combination', 'Normal', 'Dry', 'Sensitive'])

    return tone_group, undertone, skin_type

# ----------------------------------------------------------------------
# 4. ฟังก์ชันแนะนำผลิตภัณฑ์ (Recommendation Logic)
# ----------------------------------------------------------------------

def get_skincare_recommendation(user_skin_type, user_acne_score, product_db):
    """
    ปรับปรุง Logic การแนะนำ Skincare ตามคะแนนสิว (1-5) 
    เพื่อให้มีความแตกต่างระหว่างคนหน้าใสกับคนเป็นสิวเยอะ
    """
    
    # 1. กำหนดเป้าหมายหลักตามคะแนนสิว
    if user_acne_score <= 1:
        # หน้าใส/ไม่มีสิว (เน้นบำรุงและป้องกัน)
        target_ingredients = ['Ceramide', 'Hyaluronic Acid', 'Vitamin C', 'SPF50+']
        recommendation_text = "**ผิวสวยใส** สกินแคร์ควรเน้นการบำรุง เติมความชุ่มชื้น และป้องกันผิวจากแสงแดดเป็นหลัก (Sunscreen/Moisturizer)"
    
    elif user_acne_score == 2:
        # สิวเล็กน้อย (เน้นป้องกันและจัดการสิวอ่อนโยน)
        target_ingredients = ['Salicylic Acid|BHA', 'Centella Asiatica', 'Lightweight', 'Gel']
        recommendation_text = "**สิวเล็กน้อย** ควรใช้ Cleanser/Toner ที่มี BHA อ่อนโยน และเพิ่ม Spot Treatment หากจำเป็น (Treatment/Cleanser)"
    
    elif user_acne_score == 3:
        # สิวปานกลาง/อักเสบ (เน้นการรักษาที่เข้มข้นขึ้น)
        target_ingredients = ['Benzoyl Peroxide', 'Salicylic Acid|BHA', 'Retinol|Retinal', 'Oil Control']
        recommendation_text = "**สิวปานกลาง** ควรเน้น Treatment ที่มีสารรักษาสิวเข้มข้น และมอยส์เจอไรเซอร์สูตรอ่อนโยน เพื่อไม่ให้อุดตัน (Treatment/Moisturizer)"

    elif user_acne_score >= 4:
        # สิวรุนแรง (เน้นการรักษาเข้มข้นและบรรเทา)
        target_ingredients = ['Retinol|Retinal', 'Benzoyl Peroxide', 'Soothes', 'Emulsion']
        recommendation_text = "**สิวอักเสบรุนแรง** ควรปรึกษาแพทย์ผิวหนัง ควบคู่กับการใช้ผลิตภัณฑ์ที่เน้นการรักษาสิว และบรรเทาอาการอักเสบ (Treatment/Emulsion)"
        
    else:
        target_ingredients = ['Hyaluronic Acid', 'Glycerin']
        recommendation_text = "ไม่สามารถประเมินสิวได้ แต่เน้นความชุ่มชื้นไว้ก่อน"
        
    # 2. กรองผลิตภัณฑ์ตามส่วนผสมหลัก (กรองจากฐานข้อมูล)
    filtered_products = product_db[
        product_db['Key_Ingredient'].str.contains('|'.join(target_ingredients), case=False, na=False)
    ]
    
    # 3. ปรับการกรองตามประเภทผิว (เดิม)
    if user_skin_type in ['Oily', 'Combination']:
        filtered_products = filtered_products[~filtered_products['Product_Name'].str.contains('Oil|Balm', case=False)]
    elif user_skin_type in ['Dry', 'Sensitive']:
        filtered_products = filtered_products[filtered_products['Category'].isin(['Moisturizer', 'Cleanser', 'Sunscreen'])]
        
    return filtered_products.head(5), recommendation_text


def get_foundation_recommendation(user_undertone, shade_db):
    """แนะนำ Foundation ตาม Undertone"""
    # กรองตาม Undertone ที่ทำนายได้
    filtered_shades = shade_db[shade_db['Undertone'] == user_undertone]
    return filtered_shades.head(5)

def get_makeup_recommendation(user_undertone, makeup_db):
    """แนะนำ Makeup ตาม Undertone"""
    # กรอง Makeup (Blush/Lip/Contour) ตาม Tone_Type
    filtered_makeup = makeup_db[makeup_db['Tone_Type'].str.contains(user_undertone.split('-')[0], case=False, na=False)]
    return filtered_makeup.head(5)


# ----------------------------------------------------------------------
# 5. UI และการแสดงผล (Streamlit UI)
# ----------------------------------------------------------------------

st.title("AI Skincare & Makeup Advisor: Image-Only Face Analysis (DNN) 🧖‍♀️💄")
st.markdown("ระบบวิเคราะห์โทนสีผิวและลักษณะใบหน้า เพื่อแนะนำผลิตภัณฑ์ที่เหมาะสม **(เน้นเฉพาะใบหน้าส่วนที่ Crop แล้ว)**")

# ตัวเลือกจำลองสำหรับผู้ใช้
st.sidebar.header("⚙️ ข้อมูล Input จำลอง")

# ให้ผู้ใช้กรอกคะแนนสิว (สำคัญมากสำหรับการทดสอบ Logic ใหม่)
user_acne_score = st.sidebar.slider(
    "คะแนนปัญหาสิว: (1=ไม่มี, 5=รุนแรง)", 
    min_value=1, max_value=5, value=2, step=1
)
#st.sidebar.info(f"ระบบจะใช้คะแนนสิว {user_acne_score} ในการเลือกส่วนผสม")

# Upload File Section
st.subheader("อัปโหลดรูปภาพใบหน้าของคุณ 📸")
uploaded_file = st.file_uploader(
    "เคล็ดลับ: เพื่อผลลัพธ์ที่ดีที่สุด ควรใช้ภาพที่เห็นใบหน้าเต็มชัดเจนและอยู่ในแสงธรรมชาติ",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    if DNN_FACE_DETECTOR is not None:
        
        # 1. วิเคราะห์ภาพ
        cropped_face = analyze_and_crop_face(uploaded_file, DNN_FACE_DETECTOR)

        if cropped_face is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ ใบหน้าส่วนที่ Crop")
                # แสดงผลใบหน้าที่ Crop แล้ว
                st.image(cropped_face, caption="ใบหน้าที่ถูก Crop เพื่อวิเคราะห์", use_column_width=True)
                
            # 2. ทำนายโทนสีผิวและประเภทผิว
            tone_group, undertone, skin_type = predict_skin_tone_and_type(cropped_face)

            with col2:
                st.subheader("✨ ผลการวิเคราะห์ผิว")
                st.markdown(f"**โทนสีผิวหลัก (Tone Group):** <span style='background-color:#ffe4b5; padding: 4px; border-radius: 5px;'>{tone_group}</span>", unsafe_allow_html=True)
                st.markdown(f"**อันเดอร์โทน (Undertone):** <span style='background-color:#add8e6; padding: 4px; border-radius: 5px;'>{undertone}</span>", unsafe_allow_html=True)
                st.markdown(f"**ประเภทผิว (Skin Type):** <span style='background-color:#90ee90; padding: 4px; border-radius: 5px;'>{skin_type}</span>", unsafe_allow_html=True)
                st.markdown(f"**คะแนนปัญหาสิว:** <span style='background-color:#f08080; color:white; padding: 4px; border-radius: 5px;'>{user_acne_score} (จาก {st.session_state.user_acne_score if 'user_acne_score' in st.session_state else 2} ที่เลือก)</span>", unsafe_allow_html=True)


            st.markdown("---")
            
            # 3. การแนะนำผลิตภัณฑ์
            st.subheader("🛒 ผลิตภัณฑ์แนะนำสำหรับคุณ")
            
            # 3.1 Skincare Recommendation (ใช้ Logic สิวใหม่)
            skincare_recs, skincare_text = get_skincare_recommendation(skin_type, user_acne_score, PRODUCT_DB)
            st.markdown(f"#### 🧴 Skincare Recommendation: {skincare_text}")
            st.dataframe(skincare_recs[['Product_Name', 'Brand', 'Category', 'Key_Ingredient', 'Price_Range']], hide_index=True, use_container_width=True)
            
            st.markdown("---")
            
            # 3.2 Makeup & Foundation Recommendation
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