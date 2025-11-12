import streamlit as st
import pandas as pd
import numpy as np
import cv2 
import os

# ----------------------------------------------------------------------
# 0. การตั้งค่าและการโหลดฐานข้อมูล
# ----------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="AI Skincare & Makeup Advisor: Image-Only Face Analysis")

@st.cache_data
def load_db(file_path):
    """โหลดฐานข้อมูล CSV อย่างปลอดภัยและจัดการค่า Null/ประเภทข้อมูล"""
    if not os.path.exists(file_path):
        st.error(f"❗ ข้อผิดพลาดร้ายแรง: ไม่พบไฟล์ '{file_path}' กรุณาสร้างไฟล์และใส่ข้อมูล")
        return pd.DataFrame()
    
    try:
        db = pd.read_csv(file_path, na_values=['N/A', '', ' '])
        
        if 'Depth_Scale' in db.columns:
            db['Depth_Scale'] = pd.to_numeric(db['Depth_Scale'], errors='coerce')
        if 'Key_Ingredient' in db.columns:
             db['Key_Ingredient'] = db['Key_Ingredient'].astype(str).fillna('ไม่ระบุ') 
        if 'Price_Range' in db.columns:
            db['Price_Range'] = db['Price_Range'].fillna('ไม่ระบุ')
            
        if db.empty:
            st.warning(f"ไฟล์ '{file_path}' โหลดได้ แต่ไม่มีข้อมูลอยู่ภายใน")

        return db
    except Exception as e:
        st.error(f"❗ ข้อผิดพลาดในการอ่านไฟล์ {file_path}: {e}")
        return pd.DataFrame()

# โหลดฐานข้อมูลทั้งหมด
PRODUCT_DB = load_db('products.csv')
SHADE_DB = load_db('foundation_shades.csv')
TONE_DB = load_db('skin_tones.csv')
MAKEUP_DB = load_db('makeup_products.csv') # แก้ NameError แล้ว

# ----------------------------------------------------------------------
# โหลด DNN (Deep Learning) Model สำหรับ Face Detection (SSD)
# ----------------------------------------------------------------------
PROTOTXT = 'deploy.prototxt'
CAFFEMODEL = 'res10_300x300_ssd_iter_140000.caffemodel'
CONFIDENCE_THRESHOLD = 0.7 # ตั้งค่าความมั่นใจในการตรวจจับ (70%)

DNN_FACE_DETECTOR = None
if not os.path.exists(PROTOTXT) or not os.path.exists(CAFFEMODEL):
    st.error(f"❗ ไม่พบไฟล์โมเดล DNN: '{PROTOTXT}' หรือ '{CAFFEMODEL}' กรุณาวางไฟล์ใน Root Directory")
else:
    try:
        # โหลดโมเดล
        DNN_FACE_DETECTOR = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
    except Exception as e:
        st.error(f"❗ ข้อผิดพลาดในการโหลดโมเดล DNN: {e} (อาจเป็นปัญหา System Library) - จะไม่ทำการ Crop ภาพ")
        
# ----------------------------------------------------------------------
# 1. ฟังก์ชันวิเคราะห์สภาพผิวและโทนสีผิวจากภาพ (แกนหลัก)
# ----------------------------------------------------------------------

def analyze_skin_color(image):
    """วิเคราะห์โทนสีและความเข้มของผิวจากภาพที่ถูกตัดเฉพาะส่วนใบหน้าแล้ว พร้อมวิเคราะห์ความรุนแรงของสิวที่ละเอียดขึ้น"""
    if image is None or image.size == 0:
        return None 
        
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # คำนวณค่าเฉลี่ย BGR และ HSV ทั่วทั้งภาพ
    avg_bgr = np.mean(image, axis=(0, 1))
    avg_hsv = np.mean(hsv_image, axis=(0, 1))
    
    # --- 1. การคำนวณพื้นฐาน (Depth และ Undertone) ---
    V = avg_hsv[2]
    depth_scale = 1.0 + (255 - V) / 25.5 
    depth_scale = np.clip(depth_scale, 1.0, 9.0)
    
    R, G, B = avg_bgr[2], avg_bgr[1], avg_bgr[0]

    if (R > G * 1.05 and G > B * 1.05) or (R + G) / 2 > B * 1.1:
        undertone = 'Warm' 
    elif B > R * 1.05 and B > G * 1.05:
        undertone = 'Cool' 
    else:
        undertone = 'Neutral' 
        # ใช้อัตราส่วนรอยแดงเทียบกับพื้นที่ใบหน้าจริง
    if non_black_pixels > 0:
        redness_ratio = (red_pixels_count / non_black_pixels) * 100 
    else:
        redness_ratio = 0
    
    # START: การกำหนดระดับความรุนแรงของสิว/รอยแดง (Logic ใหม่)
    if redness_ratio < 0.4:
        acne_severity = 'Healthy (ผิวสุขภาพดี/ไม่มีสิว)'
    elif 0.4 <= redness_ratio < 1.5:
        # อาจเป็นรอยแดงเล็กน้อย หรือสิวอุดตันที่เริ่มอักเสบ
        acne_severity = 'Minor Redness (รอยแดงเล็กน้อย/ระคายเคือง)'
    elif 1.5 <= redness_ratio < 3.5:
        # Mild: สิวอักเสบจำนวนไม่มาก
        acne_severity = 'Mild Acne (สิวอักเสบเล็กน้อย/เริ่มมีอาการ)'
    elif 3.5 <= redness_ratio < 6.0:
        # Moderate: สิวอักเสบขนาดปานกลาง จำนวนมากขึ้น
        acne_severity = 'Moderate Acne (สิวอักเสบปานกลาง/มีรอยชัดเจน)'
    else:
        # Severe: มีสิวอักเสบขนาดใหญ่ หรือสิวเห่อเต็มใบหน้า
        acne_severity = 'Severe Acne (สิวอักเสบรุนแรง/สิวเห่อ)'
        
    # ปรับระดับความรุนแรงสำหรับผิวแห้ง (อาจเกิดจากความแห้ง/ระคายเคือง)
    if skin_type == 'Dry' and redness_ratio < 3.5:
        if acne_severity in ['Minor Redness (รอยแดงเล็กน้อย/ระคายเคือง)', 'Mild Acne (สิวอักเสบเล็กน้อย/เริ่มมีอาการ)']:
             acne_severity = 'Irritation/Low Acne (ระคายเคืองจากความแห้ง)'
    # END: การกำหนดระดับความรุนแรงของสิว/รอยแดง (Logic ใหม่)
        
    results = {
        'Skin_Type': skin_type,  
        'Acne_Severity': acne_severity,
        'Undertone': undertone,
        'Depth_Scale': float(depth_scale)
    }
    
    return results
    # --- 2. การกำหนดประเภทผิว (Skin Type) ---
    S = avg_hsv[1]
    
    if S < 100 and depth_scale < 5.0:
        skin_type = 'Dry' 
    elif depth_scale > 5.5 and S > 130:
        skin_type = 'Oily'
    else:
        skin_type = 'Combination'
        
    # --- 3. การวิเคราะห์ความรุนแรงของสิว (Acne Severity) ที่ปรับปรุงใหม่ ---
    
    # การตรวจจับรอยแดง (Redness Detection) โดยใช้ HSV (ช่วงสีแดง)
    
    # ช่วงสีแดงอ่อน (H=0 ถึง H=10)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    # ช่วงสีแดงเข้ม (H=160 ถึง H=180)
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # สร้าง mask สำหรับรอยแดง
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    redness_mask = mask1 + mask2
    
    # คำนวณเปอร์เซ็นต์ของพื้นที่ผิวที่มีรอยแดง
    # นับเฉพาะส่วนที่ไม่ใช่พื้นหลังสีดำ (ในภาพวงกลม)
    non_black_pixels = np.sum(np.any(image != [0, 0, 0], axis=2)) 
    red_pixels_count = np.sum(redness_mask > 0)
    
    # ใช้อัตราส่วนรอยแดงเทียบกับพื้นที่ใบหน้าจริง
    if non_black_pixels > 0:
        redness_ratio = (red_pixels_count / non_black_pixels) * 100 
    else:
        redness_ratio = 0
    
    # การกำหนดระดับความรุนแรงตามเปอร์เซ็นต์รอยแดง
    if redness_ratio < 0.5:
        acne_severity = 'None/Very Low (ไม่มีรอยแดงชัดเจน)'
    elif 0.5 <= redness_ratio < 2.0:
        acne_severity = 'Mild (รอยแดง/สิวเล็กน้อย)'
    elif 2.0 <= redness_ratio < 4.5:
        acne_severity = 'Moderate (มีการอักเสบ/รอยแดงชัดเจน)'
    else:
        acne_severity = 'Severe (มีการอักเสบรุนแรง/รอยแดงมาก)'
        
    # ปรับลดระดับความรุนแรงสำหรับผิวแห้ง (อาจเกิดจากความแห้ง/ระคายเคือง)
    if skin_type == 'Dry' and redness_ratio < 3.0:
        acne_severity = 'Low (ระคายเคืองจากความแห้ง)'
        
    results = {
        'Skin_Type': skin_type,  
        'Acne_Severity': acne_severity,
        'Undertone': undertone,
        'Depth_Scale': float(depth_scale)
    }
    
    return results

def process_and_analyze_image(image):
    """ตรวจจับใบหน้าด้วย DNN, ตัดภาพเฉพาะใบหน้า **เป็นวงกลม**, และส่งไปวิเคราะห์สี"""
    if image is None or DNN_FACE_DETECTOR is None:
        st.warning("⚠️ การตรวจจับใบหน้าถูกข้ามไป (โมเดล DNN โหลดไม่สำเร็จ) ประมวลผลภาพเต็ม")
        results = analyze_skin_color(image)
        return results, image

    (h, w) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    DNN_FACE_DETECTOR.setInput(blob)
    detections = DNN_FACE_DETECTOR.forward()
    
    best_face_box = None
    max_confidence = 0.0
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD and confidence > max_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_face_box = box.astype("int")
            max_confidence = confidence

    if best_face_box is None:
        st.warning("⚠️ ไม่พบใบหน้าในภาพ (DNN Detection) กรุณาอัปโหลดภาพที่เห็นใบหน้าชัดเจน")
        results = analyze_skin_color(image)
        return results, image 
    
    x1, y1, x2, y2 = best_face_box
    
    # ขยายขอบเขตใบหน้าเล็กน้อย
    w_face = x2 - x1
    h_face = y2 - y1
    expand_w = int(w_face * 0.2) 
    expand_h = int(h_face * 0.2) 
    
    x1 = max(0, x1 - expand_w)
    y1 = max(0, y1 - expand_h)
    x2 = min(w, x2 + expand_w)
    y2 = min(h, y2 + expand_h)

    # 1. Crop ภาพส่วนใบหน้าออกมาเป็นสี่เหลี่ยมก่อน
    cropped_face_image = image[y1:y2, x1:x2]
    
    # **โค้ดแก้ไข Error 0x0/cv2.error**
    if cropped_face_image.shape[0] == 0 or cropped_face_image.shape[1] == 0:
        st.warning("⚠️ ข้อผิดพลาดในการ Crop: ขนาดใบหน้าไม่ถูกต้อง หรืออยู่นอกขอบเขตภาพ")
        results = analyze_skin_color(image)
        return results, image
    
    # 2. สร้าง Mask วงกลมบนภาพที่ Crop แล้ว
    (ch, cw) = cropped_face_image.shape[:2]
    center = (cw // 2, ch // 2)
    radius = min(cw, ch) // 2 
    
    # สร้าง mask 3 channel สำหรับภาพสี (BGR)
    circle_mask = np.zeros(cropped_face_image.shape, dtype=np.uint8) 
    
    # วาดวงกลมสีขาวบน mask
    cv2.circle(circle_mask, center, radius, (255, 255, 255), -1) 
    
    # 3. นำ mask มาซ้อนทับ: ภาพที่ Crop เฉพาะวงกลม (พื้นหลังจะกลายเป็นสีดำ)
    circular_face = cv2.bitwise_and(cropped_face_image, circle_mask)
        
    results = analyze_skin_color(circular_face) # วิเคราะห์จากภาพวงกลม
    return results, circular_face # คืนค่าผลลัพธ์ และภาพใบหน้าที่ถูก Crop เป็นวงกลม


# ----------------------------------------------------------------------
# 2. ฟังก์ชันแนะนำผลิตภัณฑ์ (Rule-Based Logic)
# ----------------------------------------------------------------------

def recommend_skincare(skin_analysis_results, db):
    """กำหนดกฎเกณฑ์การแนะนำผลิตภัณฑ์บำรุงผิว"""
    skin_type = skin_analysis_results['Skin_Type'].split(' ')[0] # ใช้คำแรก ('Dry', 'Oily', 'Combination')
    acne_severity = skin_analysis_results['Acne_Severity'].split(' ')[0] # ใช้คำแรก ('None/Very', 'Mild', 'Moderate', 'Severe', 'Low')
    recommendations = {}

    # 1. Cleanser
    if skin_type in ['Oily', 'Combination'] and acne_severity not in ['None/Very', 'Low']:
        reco = db[(db['Category'] == 'Cleanser') & (db['Key_Ingredient'].str.contains('Salicylic Acid|BHA', case=False, na=False))].head(1)
    elif skin_type == 'Dry':
        reco = db[(db['Category'] == 'Cleanser') & (db['Key_Ingredient'].str.contains('Ceramide|Glycerin', case=False, na=False))].head(1)
    else:
        reco = db[db['Category'] == 'Cleanser'].head(1)
    if not reco.empty:
        recommendations['Step 1: Cleanser (ทำความสะอาด)'] = reco

    # 2. Treatment
    if acne_severity in ['Moderate', 'Severe']:
        reco = db[(db['Key_Ingredient'].str.contains('Niacinamide|Salicylic Acid|Benzoyl Peroxide', case=False, na=False)) & (db['Category'] == 'Treatment')].head(2)
        if not reco.empty:
            recommendations['Step 2: Targeted Treatment (รักษาสิว/ลดรอย)'] = reco
    
    # 3. Moisturizer
    if skin_type in ['Oily', 'Combination']:
        reco = db[(db['Category'] == 'Moisturizer') & (db['Product_Name'].str.contains('Gel|Water|Lightweight', case=False, na=False))].head(1)
    else:
        reco = db[(db['Category'] == 'Moisturizer') & (db['Key_Ingredient'].str.contains('Ceramide|Squalane', case=False, na=False))].head(1)
    if not reco.empty:
        recommendations['Step 3: Moisturizer (เติมความชุ่มชื้น)'] = reco

    # 4. Sunscreen
    reco = db[db['Category'] == 'Sunscreen'].head(1)
    if not reco.empty:
        recommendations['Step 4: Sunscreen (ป้องกัน)'] = reco
        
    return recommendations


def recommend_foundation(undertone, depth_scale, db):
    """จับคู่เฉดสีรองพื้นตามโทนผิวและความเข้ม"""
    if db.empty:
        return pd.DataFrame()
        
    filtered_df = db[db['Depth_Scale'].notna()]
    filtered_df = filtered_df[filtered_df['Undertone'] == undertone]
    
    if filtered_df.empty:
        filtered_df = db[db['Undertone'] == 'Neutral']

    if filtered_df.empty:
        return pd.DataFrame() 

    filtered_df['Depth_Diff'] = np.abs(filtered_df['Depth_Scale'] - depth_scale)
    
    return filtered_df.sort_values(by='Depth_Diff').head(3).drop(columns=['Depth_Diff']).reset_index(drop=True)


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
    if PRODUCT_DB.empty or SHADE_DB.empty or TONE_DB.empty or MAKEUP_DB.empty:
        st.error("❗ โครงสร้างฐานข้อมูลไม่สมบูรณ์ โปรดตรวจสอบไฟล์ CSV ทั้ง 4 ไฟล์ว่ามีข้อมูลอยู่หรือไม่")
        return

    st.title("🔬 AI Skincare & Makeup Advisor: Image-Only Face Analysis (DNN)")
    st.caption("ระบบวิเคราะห์โทนสีผิว ความเข้มผิว และการแนะนำผลิตภัณฑ์จากภาพ (เน้นเฉพาะใบหน้าที่ Crop แล้ว)")
    st.markdown("---")
    
    st.subheader("อัปโหลดรูปภาพใบหน้าของคุณ")
    st.info("💡 **เคล็ดลับ:** เพื่อผลลัพธ์ที่แม่นยำ ควรใช้ภาพที่เห็นใบหน้าชัดเจน และอยู่ในแสงธรรมชาติ")
    
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    results = None
    cropped_face_for_display = None 
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col2:
            st.subheader("📊 ผลการวิเคราะห์สภาพผิวและโทนสี")
            with st.spinner(f'กำลังประมวลผลรูปภาพโดยใช้ DNN Face Detection...'):
                uploaded_file.seek(0)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                
                # **เรียกใช้ฟังก์ชันหลักที่ทำการ Crop และ Analyze**
                results, cropped_face_for_display = process_and_analyze_image(image)
            
            if results:
                st.success("✅ วิเคราะห์สำเร็จ!")
                # แสดงผลลัพธ์ที่ปรับปรุงใหม่
                st.metric(label="ประเภทผิวหลัก", value=f"**{results['Skin_Type']}**")
                st.info(f"**โทนสีผิว (Undertone):** {results['Undertone']} | **ระดับความเข้ม (Depth):** {results['Depth_Scale']:.2f}")
                st.metric(label="ระดับความรุนแรงของสิว/รอยแดง", value=f"**{results['Acne_Severity']}**")
            else:
                st.warning("⚠️ ไม่สามารถวิเคราะห์ได้ (ไม่พบใบหน้า หรือไฟล์มีปัญหา)")

        with col1:
            if cropped_face_for_display is not None and cropped_face_for_display.size > 0:
                # ตรวจสอบว่าภาพที่แสดงคือภาพที่ถูก Crop แล้วจริงๆ หรือไม่
                if cropped_face_for_display.shape == image.shape:
                    caption_text = "ภาพเต็ม (ไม่พบใบหน้า หรือ DNN โหลดไม่ได้)"
                else:
                    caption_text = "ใบหน้าที่ระบบใช้ในการวิเคราะห์ (Crop เป็นวงกลม)"
                    
                st.image(cv2.cvtColor(cropped_face_for_display, cv2.COLOR_BGR2RGB), caption=caption_text, use_column_width=True)
            else:
                st.caption(f"กำลังรอผลลัพธ์การประมวลผลรูปภาพ หรือไม่พบใบหน้า")

        st.markdown("---")
        
        # 5. แสดงผลการแนะนำ (ถ้ามี results)
        if results:
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
                st.markdown(f"**เฉดสีที่ใกล้เคียงที่สุด** สำหรับโทน **{results['Undertone']}** และผิวระดับ **{results['Depth_Scale']:.2f}**:")
                
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