import cv2
import os

# Video yolu
path = "video/video.mp4"

# Video yolunun varlığını kontrol et
if not os.path.exists(path):
    print(f"Hata: '{path}' dosyası bulunamadı!")
    exit()

# Video dosyasını aç
cap = cv2.VideoCapture(path)

# Video dosyasının başarıyla açıldığını kontrol et
if not cap.isOpened():
    print("Hata: Video dosyası açılamadı!")
    exit()

# Video FPS'sini al ve uygun gecikmeyi hesapla
fps = int(cap.get(cv2.CAP_PROP_FPS))
delay = int(1000 / fps)

# Videoyu oynatma
while True:
    ref, frame = cap.read()
    if not ref:
        print("Uyarı: Çerçeve okunamadı veya video bitti.")
        break

    # Çerçevenin boyutlarını küçült (640x480 çözünürlük)
    frame_resized = cv2.resize(frame, (640, 480))

    # Küçültülmüş çerçeveyi göster
    cv2.imshow("frame", frame_resized)

    # Çıkış için 'q' tuşuna basılmasını bekle
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
