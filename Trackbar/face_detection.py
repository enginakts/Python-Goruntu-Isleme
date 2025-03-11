import cv2  
import time

# Kamera erişimini başlat
video_kaynak = cv2.VideoCapture(0)

# Haarcascade modellerini yükle
cascade_goz = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
cascade_yuz = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cascade_gulucuk = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# FPS hesaplamak için zamanlayıcı başlat
baslangic_zamani = time.time()
kare_sayisi = 0

while True:
    # Kameradan görüntü oku
    basarili, kare = video_kaynak.read()
    if not basarili:
        break  # Kamera erişimi başarısızsa döngüden çık
    
    kare_sayisi += 1
    
    # Görüntüyü gri seviyeye çevir (Haarcascade daha iyi çalışır)
    gri_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    
    # Yüzleri tespit et
    yuzler = cascade_yuz.detectMultiScale(gri_kare, scaleFactor=1.1, minNeighbors=9)
    
    for (x, y, genislik, yukseklik) in yuzler:
        # Yüzü dikdörtgen içine al
        cv2.rectangle(kare, (x, y), (x + genislik, y + yukseklik), (0, 255, 0), 2)
        
        # Yüz bölgesini belirle
        gri_yuz_bolgesi = gri_kare[y: y + yukseklik, x:x + genislik]
        renkli_yuz_bolgesi = kare[y: y + yukseklik, x:x + genislik]
        
        # Gözleri tespit et
        gozler = cascade_goz.detectMultiScale(gri_yuz_bolgesi, scaleFactor=1.1, minNeighbors=9)
        
        for (gx, gy, g_w, g_h) in gozler:
            cv2.rectangle(renkli_yuz_bolgesi, (gx, gy), (gx + g_w, gy + g_h), (255, 0, 0), 2)
        
        # Gülümsemeyi tespit et
        gulucukler = cascade_gulucuk.detectMultiScale(gri_yuz_bolgesi, scaleFactor=1.7, minNeighbors=22)
        
        for (gx, gy, g_w, g_h) in gulucukler:
            cv2.rectangle(renkli_yuz_bolgesi, (gx, gy), (gx + g_w, gy + g_h), (0, 0, 255), 2)
    
    # FPS hesapla ve ekrana yazdır
    gecen_zaman = time.time() - baslangic_zamani
    fps = kare_sayisi / gecen_zaman
    cv2.putText(kare, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Görüntüyü göster
    cv2.imshow("Canlı Yüz Tespiti", kare)
    
    # 'q' tuşuna basılınca çık
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak ve pencereleri kapat
video_kaynak.release()
cv2.destroyAllWindows()
