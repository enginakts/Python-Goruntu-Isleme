import cv2
import numpy as np
from tracker import EuclideanDistTracker  # Sadece bu satırı değiştirin

# Video dosyasını yükle
cap = cv2.VideoCapture('Video/test.mp4')
# Araç tespiti için cascade sınıflandırıcısını yükle
car_cascade = cv2.CascadeClassifier('Cars.xml')

# Tracker oluştur
tracker = EuclideanDistTracker()

# Arka plan çıkarıcı
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Sayaç değişkenleri
total_cars = 0
cars_in_frame = 0

# Tespit çizgisi koordinatları
line_position = 240  # Ekranın ortası
detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü yeniden boyutlandır
    frame = cv2.resize(frame, (640, 480))
    
    # İlgi alanını belirle (ROI)
    roi = frame[200:480, 0:640]
    
    # Maske oluştur
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    
    # Konturları bul
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Minimum alan filtreleme
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])
    
    # Tracker'ı güncelle
    boxes_ids = tracker.update(detections)
    
    # Tespit çizgisini çiz
    cv2.line(frame, (0, line_position), (640, line_position), (0, 255, 0), 2)
    
    cars_in_frame = 0
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y + 200), (x + w, y + h + 200), (0, 255, 0), 2)
        
        # Araç sayma mantığı
        center_y = y + h // 2
        if line_position - 10 <= center_y + 200 <= line_position + 10:
            total_cars += 1
        
        cars_in_frame += 1

    # Bilgileri ekrana yazdır
    cv2.putText(frame, f"Anlik Arac Sayisi: {cars_in_frame}", (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.putText(frame, f"Toplam Gecen Arac: {total_cars}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # Sonucu göster
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
