import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2
import easyocr
import numpy as np

# Görüntü dosyasının yolunu belirle
image_path = "Images/plaka.jpg"

# Görüntüyü oku
img = cv2.imread(image_path)

# Görüntünün başarıyla okunup okunmadığını kontrol et
if img is None:
    print("Hata: Görüntü okunamadı!")
    exit()

# EasyOCR okuyucusunu başlat (İngilizce dili için, GPU kullanmadan)
reader = easyocr.Reader(["en"], gpu=False)

# Görüntüdeki metinleri tespit et
text = reader.readtext(img)

# Güven eşiği - bu değerin üzerindeki tespitler kabul edilecek
threshold = 0.25

# Tespit edilen her metin için işlem yap
for t_, t in enumerate(text):
    print(f"Tespit {t_+1}:", t)

    # Tespit edilen metnin koordinatlarını, içeriğini ve güven skorunu al
    bbox, text, score = t
    
    # Güven skoru eşik değerinden yüksekse işleme devam et
    if score > threshold:
        # Sınırlayıcı kutunun köşe koordinatlarını tam sayıya çevir
        (x1, y1) = (int(bbox[0][0]), int(bbox[0][1]))  # Sol üst köşe
        (x2, y2) = (int(bbox[2][0]), int(bbox[2][1]))  # Sağ alt köşe
        
        # Tespit edilen metin etrafına kırmızı dikdörtgen çiz
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
        
        # Tespit edilen metni dikdörtgenin üstüne yeşil renkle yaz
        cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Görüntüyü göster
cv2.imshow("Tespit Edilen Metin", img)
cv2.waitKey(0)  # Herhangi bir tuşa basılana kadar bekle
cv2.destroyAllWindows()  # Pencereyi kapat
