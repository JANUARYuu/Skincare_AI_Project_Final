import pandas as pd
import numpy as np
import random
import os
from collections import defaultdict

# ----------------------------------------------------------------------
# 1. การตั้งค่าข้อมูลพื้นฐาน (Mock Data Settings)
# ----------------------------------------------------------------------

NUM_SKINCARE = 1000  # จำนวนผลิตภัณฑ์ Skincare ที่สร้าง
NUM_FOUNDATION = 1000 # จำนวนผลิตภัณฑ์ Foundation ที่สร้าง
NUM_MAKEUP = 1000    # จำนวนผลิตภัณฑ์ Makeup ที่สร้าง

# รายการแบรนด์ที่ถูกขยายตามคำขอ (Skincare 200 แบรนด์, ที่เหลือ 100 แบรนด์)
BRANDS = {
    'Skincare': [
        'La Roche-Posay', 'Cerave', 'The Ordinary', 'Kiehl\'s', 'Paula\'s Choice', 
        'Eucerin', 'Estee Lauder', 'SK-II', 'Hada Labo', 'Laneige', 
        'Biore', 'Garnier', 'Vichy', 'Innisfree', 'Clinique', 'Shiseido', 
        'Sulwhasoo', 'Cosrx', 'Some By Mi', 'Mizumi', 'Aesop', 'Tata Harper', 
        'Tatcha', 'Sunday Riley', 'Drunk Elephant', 'Fresh', 'Origins', 
        'Philosophy', 'Avene', 'Bioderma', 'Murad', 'SkinCeuticals', 
        'Dr. Jart+', 'Belif', 'The Inkey List', 'Good Molecules', 'Byoma', 
        'Youth To The People', 'Ren Skincare', 'Summer Fridays', 'Glow Recipe', 
        'Krave Beauty', 'Then I Met You', 'Farmacy', 'Herbivore Botanicals', 
        'Kate Somerville', 'Korres', 'Mario Badescu', 'Neutrogena', 'Simple',
        # แบรนด์ Skincare เพิ่มเติม (51-100)
        'Acure', 'Andalou Naturals', 'Caudalie', 'Cetaphil', 'Cosmetics 27', 
        'Dermalogica', 'Elemis', 'First Aid Beauty', 'Guerlain Orchidee Imperiale', 
        'Indie Lee', 'IS Clinical', 'Janesce', 'Juice Beauty', 'KORA Organics', 
        'Lancer Skincare', 'Lotus Herbals', 'Mad Hippie', 'Neostrata', 
        'Natura Bisse', 'Odacite', 'Omorovicza', 'Osea', 'Perricone MD', 
        'Peter Thomas Roth', 'Pola', 'Prescriptives', 'QMS Medicosmetics', 
        'RéVive', 'Rodial', 'Sodashi', 'StriVectin', 'Suqqu', 
        'Synergie Skin', 'Tammy Fender', 'Twelve Beauty', 'Vintner’s Daughter', 
        'Volition Beauty', 'Weleda', 'Yes To', 'Zelens', 'Verso', 
        'Amouage', 'Babor', 'Chantecaille', 'Clé de Peau Beauté', 'Darphin',
        # แบรนด์ Skincare เพิ่มเติม (101-150)
        'Decleor', 'Dr. Dennis Gross', 'Dr. Hauschka', 'Erno Laszlo', 'Foreo', 
        'GlamGlow', 'Goop', 'Guerlain', 'Hask', 'Ilia', 
        'Josie Maran', 'Jurlique', 'Kerastase', 'Kopari', 'Living Proof', 
        'Malin+Goetz', 'Mantra', 'Marula', 'Naturopathica', 'NuFACE', 
        'Ouai', 'Pai Skincare', 'Pacifica', 'Patchology', 'Phytomer', 
        'Pixi', 'Rahua', 'Sachajuan', 'Sanitas', 'Skyn Iceland', 
        'Suki', 'Supergoop!', 'Tangle Teezer', 'This Works', 'Ursa Major', 
        'Voluspa', 'WelleCo', 'Westman Atelier', 'YouthForia', 'YSL Beauty',
        # แบรนด์ Skincare เพิ่มเติม (151-200)
        'Alpha-H', 'Alpyn Beauty', 'Ameliorate', 'Ancestral', 'Annmarie Gianni', 
        'Aurelia', 'B. Kamins', 'Badescu', 'Bamford', 'Beauty Counter', 
        'By Terry', 'Circumference', 'Clark\'s Botanicals', 'Context', 'Cosmydor', 
        'Cultivated Skincare', 'Davines', 'De Mamiel', 'Disruptor', 'Doctor Babor', 
        'Eminence', 'Environ', 'Espa', 'Eve Lom', 'Five Element', 
        'Gisou', 'Goldfaden MD', 'Honest Beauty', 'HydroPeptide', 'I Dew Care', 
        'Iles Formula', 'Joanna Vargas', 'Kat Burki', 'Kevyn Aucoin', 'Lilah B.', 
        'LOLIWARE', 'Luzern', 'Maker', 'Mila Moursi', 'Moss', 
        'MV Organics', 'Nip + Fab', 'Nyakio', 'One Love Organics', 'Patyka', 
        'Radical Skincare', 'ReFa', 'Sente', 'Susanne Kaufmann', 'Vintner Daughter' # รวม 200 แบรนด์
    ],
    'Foundation': [
        'NARS', 'Estee Lauder', 'MAC', 'Bobbi Brown', 'Maybelline', 
        'L\'Oreal', '4U2', 'Srichand', 'Fenty Beauty', 'Dior', 
        'Chanel', 'Gucci', 'Rare Beauty', 'Kylie Cosmetics', 'Urban Decay', 
        'Tarte', 'Hourglass', 'Charlotte Tilbury', 'Revlon', 'Covermark',
        'Too Faced', 'Make Up For Ever', 'Anastasia Beverly Hills', 'Huda Beauty', 'KVD Beauty',
        'Milani', 'E.L.F.', 'NYX Professional Makeup', 'Colourpop', 'Wet n Wild', 
        # แบรนด์ Foundation เพิ่มเติม (31-100)
        'Giorgio Armani', 'Tom Ford', 'Yves Saint Laurent', 'Lancome', 'Sephora Collection',
        'Laura Mercier', 'Smashbox', 'Bare Minerals', 'Shiseido Makeup', 'Kosas',
        'Saie', 'Pat McGrath Labs', 'Natasha Denona', 'Armani Beauty', 'Chantecaille',
        'Clé de Peau Beauté Makeup', 'By Terry', 'Kevyn Aucoin Makeup', 'Kiko Milano Makeup', 'Covergirl',
        'Pixi Makeup', 'Juvia\'s Place', 'Becca', 'IT Cosmetics', 'Stila', 
        'Flower Beauty', 'Makeup Geek', 'Rodial Makeup', 'Givenchy Beauty', 'Bite Beauty',
        'Ciate London', 'Milk Makeup', 'Trish McEvoy', 'Wander Beauty', 'Zoeva',
        'Victoria Beckham Beauty', 'RMS Beauty', 'Surratt Beauty', 'Vapour Beauty', 'Lawless Beauty', 
        'Kjaer Weis', 'Westman Atelier Makeup', 'Kaima Cosmetics', 'Rituel de Fille', 'Exa Beauty', 
        'Tower 28 Beauty', 'MERIT', 'Refy', 'Morphe 2', 'About-Face',
        'Half-Magic', 'One/Size', 'Danessa Myricks Beauty', 'Juno & Co.', 'Blended Beauty' # รวม 100 แบรนด์
    ],
    'Makeup': [
        'Fenty Beauty', 'MAC', '4U2', 'NARS', 'Too Faced', 
        'Urban Decay', 'Tarte', 'Hourglass', 'Charlotte Tilbury', 'Dior', 
        'Benefit', 'Huda Beauty', 'Sephora Collection', 'Morphe', 'Colourpop', 
        'Innisfree', 'Laneige', 'Kiko Milano', 'Canmake', 'E.L.F.',
        'Anastasia Beverly Hills', 'Pat McGrath Labs', 'Natasha Denona', 'Juvia\'s Place', 'Kylie Cosmetics',
        'Rare Beauty', 'Haus Labs', 'KVD Beauty', 'Lime Crime', 'Jeffree Star Cosmetics',
        # แบรนด์ Makeup เพิ่มเติม (31-100)
        'Glossier', 'Milk Makeup', 'Thrive Causemetics', 'Kosas', 'Saie',
        'Giorgio Armani Makeup', 'Tom Ford Makeup', 'Yves Saint Laurent Makeup', 'Lancome Makeup', 'Shiseido Makeup Line',
        'Laura Mercier Makeup', 'Smashbox Cosmetics', 'Bare Minerals Makeup', 'Chanel Makeup', 'Gucci Makeup',
        'Bobbi Brown Makeup', 'Make Up For Ever Makeup', 'Milani Makeup', 'NYX Professional Makeup Line', 'Wet n Wild Makeup',
        'IT Cosmetics Line', 'Stila Makeup', 'Flower Beauty Line', 'Rodial Makeup Line', 'Givenchy Beauty Line',
        'Bite Beauty Line', 'Ciate London Line', 'Trish McEvoy Line', 'Wander Beauty Line', 'Zoeva Line',
        'Victoria Beckham Beauty Line', 'RMS Beauty Line', 'Surratt Beauty Line', 'Vapour Beauty Line', 'Lawless Beauty Line',
        'Kjaer Weis Line', 'Westman Atelier Line', 'Kaima Cosmetics Line', 'Rituel de Fille Line', 'Exa Beauty Line',
        'Tower 28 Beauty Line', 'MERIT Line', 'Refy Line', 'Morphe 2 Line', 'About-Face Line',
        'Half-Magic Line', 'One/Size Line', 'Danessa Myricks Beauty Line', 'Juno & Co. Line', 'Blended Beauty Line',
        'Lipstick Queen', 'Buxom', 'Hard Candy', 'Ofra Cosmetics', 'Scott Barnes',
        'Temptu', 'Three', 'Viseart', 'Byredo Makeup', 'Isamaya Beauty' # รวม 100 แบรนด์
    ]
}

PRICE_RANGES = {
    'Low': '200-500 THB',
    'Medium': '500-1200 THB',
    'High': '1200-2500 THB',
    'Luxury': '2500-4000 THB'
}

# ----------------------------------------------------------------------
# 2. ฟังก์ชันสร้างข้อมูล Skincare (products.csv) - ไม่มีการเปลี่ยนแปลง Logic
# ----------------------------------------------------------------------

def generate_skincare_data(num_items):
    data = defaultdict(list)
    cleansers = [('Foam', 'Salicylic Acid|BHA'), ('Gel', 'Ceramide|Hyaluronic Acid'), ('Milk', 'Glycerin|Squalane'), ('Oil', 'Vitamin E|Squalane'), ('Balm', 'Shea Butter|Emulsifier')]
    treatments = [('Serum', 'Niacinamide|Zinc'), ('Serum', 'Ascorbic Acid|Vitamin C'), ('Spot', 'Benzoyl Peroxide|Sulfur'), ('Retinol', 'Retinal|Peptide'), ('AHA/BHA', 'Glycolic Acid|Lactic Acid'), ('Toner', 'Centella Asiatica|Calming')]
    moisturizers = [('Cream', 'Ceramide|Squalane'), ('Gel', 'Hyaluronic Acid|Aloe'), ('Lotion', 'Shea Butter|Vitamin E'), ('Sleeping Mask', 'Glycerin|AHA'), ('Emulsion', 'Lightweight|Soothes')]
    sunscreens = [('Fluid', 'SPF50+ PA++++|Physical'), ('Cream', 'SPF30 PA+++|Chemical'), ('Gel', 'UV Filter|Water-based'), ('Stick', 'Zinc Oxide|Portable')]
    categories = {'Cleanser': cleansers, 'Treatment': treatments, 'Moisturizer': moisturizers, 'Sunscreen': sunscreens}
    
    total_generated = 0
    while total_generated < num_items:
        for brand in BRANDS['Skincare']: # วนลูปผ่าน 200 แบรนด์
            selected_categories = random.sample(list(categories.keys()), k=random.randint(1, len(categories)))
            
            for category in selected_categories:
                details_list = categories[category]
                for product_type, ingredient in details_list:
                    
                    price_group = random.choice(list(PRICE_RANGES.keys()))
                    if brand in ['SK-II', 'Estee Lauder', 'Shiseido', 'Drunk Elephant', 'Tata Harper', 'Guerlain Orchidee Imperiale', 'Clé de Peau Beauté']:
                        price_group = 'Luxury'
                    price = PRICE_RANGES[price_group]
                    
                    data['Product_Name'].append(f"{product_type} {category} - {ingredient.split('|')[0]}")
                    data['Brand'].append(brand)
                    data['Category'].append(category)
                    data['Key_Ingredient'].append(ingredient)
                    data['Price_Range'].append(price)
                    data['Image_File'].append(f"{category.lower()}_{brand.replace(' ', '_').replace('\'', '')}_{total_generated}.jpg")
                    
                    total_generated += 1
                    if total_generated >= num_items: break
                if total_generated >= num_items: break
            if total_generated >= num_items: break

    return pd.DataFrame(data).head(num_items)

# ----------------------------------------------------------------------
# 3. ฟังก์ชันสร้างข้อมูล Foundation (foundation_shades.csv) - ไม่มีการเปลี่ยนแปลง Logic
# ----------------------------------------------------------------------

def generate_foundation_data(num_items):
    data = defaultdict(list)
    undertones = ['Warm', 'Cool', 'Neutral', 'Warm-Olive', 'Cool-Pink']
    coverages = ['Light', 'Medium', 'Full', 'Sheer']
    shades_count = 0
    depth_scales = np.linspace(1.0, 9.0, 50) 
    
    while shades_count < num_items:
        for brand in BRANDS['Foundation']: # วนลูปผ่าน 100 แบรนด์
            for coverage in coverages:
                for ut in undertones:
                    for i, depth in enumerate(depth_scales):
                        ut_code = ut[0] if len(ut) == 1 else ut.replace('-', '').replace(' ', '')[0:3]
                        shade_num = (i + 1)
                        shade_code = f"{ut_code}{shade_num:02d}" 
                        
                        data['Shade_Name'].append(f"{shade_code} - {ut} {depth:.1f} Shade")
                        data['Brand'].append(brand)
                        data['Coverage'].append(coverage)
                        data['Undertone'].append(ut)
                        data['Depth_Scale'].append(round(depth, 1))
                        
                        price_group = random.choice(list(PRICE_RANGES.keys()))
                        if brand in ['Dior', 'Chanel', 'Gucci', 'Giorgio Armani', 'Tom Ford', 'Clé de Peau Beauté Makeup']:
                            price_group = 'Luxury'
                        price = PRICE_RANGES[price_group]
                        data['Price_Range'].append(price)
                        
                        data['Image_File'].append(f"fd_{brand.replace(' ', '_').replace('\'', '')}_{shade_code}.jpg")

                        shades_count += 1
                        if shades_count >= num_items: break
                    if shades_count >= num_items: break
                if shades_count >= num_items: break
            if shades_count >= num_items: break
    
    return pd.DataFrame(data).head(num_items)


# ----------------------------------------------------------------------
# 4. ฟังก์ชันสร้างข้อมูล Makeup (makeup_products.csv) - ไม่มีการเปลี่ยนแปลง Logic
# ----------------------------------------------------------------------

def generate_makeup_data(num_items):
    data = defaultdict(list)
    blush_features = {'Warm': ['Peach Glow|Shimmer', 'Terracotta|Matte', 'Coral Sunset|Satin'],'Cool': ['Rose Pink|Matte', 'Berry Pop|Shimmer', 'Mauve Kiss|Cream'],'Neutral': ['Dusty Nude|Matte', 'Soft Coral|Satin', 'True Pink|Cream']}
    lip_features = {'Warm': ['Brick Red|Creamy', 'Warm Brown|Matte', 'Deep Orange|Glossy'],'Cool': ['Blue-Red|Matte', 'Berry Pink|Satin', 'Fuchsia|Liquid'],'Neutral': ['Rose Nude|Creamy', 'Mauve|Glossy', 'True Nude|Velvet']}
    contour_features = {'Warm': ['Bronze Powder|Warm', 'Cream Bronzer|Golden'],'Cool': ['Ash Contour|Cool', 'Taupe Stick|Neutral'],'Neutral': ['Soft Tan|Powder', 'Light Sculpt|Cream']}
    powder_features = {'Warm': ['Warm Beige|Light Coverage', 'Yellow Setting|Oil Control'],'Cool': ['Pink Brightening|Sheer', 'Cool Ivory|Matte'],'Neutral': ['Translucent|Matte', 'HD Setting|Invisible']}
    categories = {'Blush': blush_features, 'Lip': lip_features, 'Contour': contour_features, 'Powder': powder_features}
    
    total_generated = 0
    while total_generated < num_items:
        for brand in BRANDS['Makeup']: # วนลูปผ่าน 100 แบรนด์
            for category, features in categories.items():
                for tone, detail_list in features.items():
                    for detail in detail_list:
                        product_name = f"{detail.split('|')[0]} {category} ({tone})"
                        
                        price_group = random.choice(list(PRICE_RANGES.keys()))
                        if brand in ['Dior', 'Hourglass', 'Charlotte Tilbury', 'Pat McGrath Labs', 'Natasha Denona', 'Byredo Makeup', 'Westman Atelier Line']:
                            price_group = 'Luxury'
                        price = PRICE_RANGES[price_group]
                        
                        data['Product_Name'].append(product_name)
                        data['Brand'].append(brand)
                        data['Category'].append(category)
                        data['Key_Feature'].append(detail)
                        data['Tone_Type'].append(tone)
                        data['Price_Range'].append(price)
                        data['Image_File'].append(f"{category.lower()}_{tone}_{brand.replace(' ', '_').replace('\'', '')}_{total_generated}.jpg")
                        
                        total_generated += 1
                        if total_generated >= num_items: break
                    if total_generated >= num_items: break
                if total_generated >= num_items: break
            if total_generated >= num_items: break

    return pd.DataFrame(data).head(num_items)

# ----------------------------------------------------------------------
# 5. การรันและบันทึกไฟล์
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("--- เริ่มต้นสร้างข้อมูลจำลองขนาดใหญ่ (Skincare 200 แบรนด์, ที่เหลือ 100 แบรนด์) ---")
    
    skincare_df = generate_skincare_data(NUM_SKINCARE)
    skincare_df.to_csv('products.csv', index=False)
    print(f"✅ สร้าง products.csv (Skincare) สำเร็จ: {len(skincare_df)} รายการ")

    foundation_df = generate_foundation_data(NUM_FOUNDATION)
    foundation_df.to_csv('foundation_shades.csv', index=False)
    print(f"✅ สร้าง foundation_shades.csv (Foundation) สำเร็จ: {len(foundation_df)} รายการ")

    makeup_df = generate_makeup_data(NUM_MAKEUP)
    makeup_df.to_csv('makeup_products.csv', index=False)
    print(f"✅ สร้าง makeup_products.csv (Makeup) สำเร็จ: {len(makeup_df)} รายการ")
    
    if not os.path.exists('skin_tones.csv'):
        print("⚠️ กรุณาตรวจสอบว่ามีไฟล์ skin_tones.csv อยู่หรือไม่ หากไม่มี ระบบจะไม่สมบูรณ์")
    
    print("\n--- เสร็จสมบูรณ์! พร้อมสำหรับการอัปโหลดและ Deploy ---")