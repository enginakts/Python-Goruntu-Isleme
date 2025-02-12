# Gerekli kütüphanelerin import edilmesi
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import pytesseract

# Tesseract OCR'ın yolunun belirtilmesi
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    # Görüntünün okunması
    image_path = "Images/plaka.jpg"
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Görüntü okunamadı")

    # Görüntü ön işleme adımları
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) # Gürültü azaltma
    edged = cv2.Canny(bfilter, 30, 200) # Kenar tespiti
    
    # Morfolojik işlemler ile kenarları güçlendir
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edged = cv2.dilate(edged, kernel, iterations=1)
    edged = cv2.erode(edged, kernel, iterations=1)

    # Konturların gelişmiş tespiti
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = img.copy()
    cv2.drawContours(img_copy, contours, -1, (0,255,0), 3)
    
    # Plaka adaylarını filtrele
    possible_plates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Minimum alan kontrolü
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if len(approx) == 4:  # Dörtgen şekil kontrolü
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = float(w)/h
                
                # Plaka boyut oranı kontrolü (tipik plaka oranı 3.5-4.5 arası)
                if 2.5 <= aspect_ratio <= 5.0:
                    possible_plates.append((x,y,w,h,approx))

    if not possible_plates:
        raise Exception("Plaka bölgesi tespit edilemedi")

    # En uygun plaka adayını seç
    possible_plates.sort(key=lambda x: cv2.contourArea(x[4]), reverse=True)
    x,y,w,h,plate_cnt = possible_plates[0]
    
    # Plaka bölgesini kırp
    plate = gray[y:y+h, x:x+w]
    
    # Plaka görüntüsünü iyileştir
    plate = cv2.resize(plate, None, fx=1.2, fy=1.2)
    plate = cv2.GaussianBlur(plate, (5,5), 0)
    _, plate = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Karakter tanıma için görüntüyü hazırla
    kernel = np.ones((2,2), np.uint8)
    plate = cv2.dilate(plate, kernel, iterations=1)
    plate = cv2.erode(plate, kernel, iterations=1)

    # OCR işlemi
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    text = pytesseract.image_to_string(plate, lang='eng', config=custom_config)
    text = ''.join(e for e in text if e.isalnum())  # Sadece alfanumerik karakterleri al
    
    # Sonucu görüntü üzerine yaz
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Sonuçların gösterilmesi
    cv2.imshow("Plaka Tespiti", img)
    cv2.imshow("İşlenmiş Plaka", plate)
    cv2.imshow("Kenar Tespiti", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Hata oluştu: {str(e)}")
    cv2.destroyAllWindows()
