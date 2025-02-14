import cv2

# Resim yolunu doğru şekilde belirleyin
img_path = "Images/img1.jpg"

# Resmi yükle
img = cv2.imread(img_path)

# Resmin başarıyla yüklenip yüklenmediğini kontrol edin
if img is None:
    print("Resim yüklenemedi. Lütfen dosya yolunu kontrol edin.")
else:
    # Yazı tipi
    font = cv2.FONT_HERSHEY_COMPLEX

    # Görüntü üzerine metin yazdır
    cv2.putText(img, 
                "Bu bir resimdir", 
                (50, 100),  # Metin başlangıç konumu (x, y)
                font, 
                1.5,  # Yazı boyutu
                (255, 0, 0),  # Yazı rengi (BGR formatında mavi renk)
                2,  # Kalınlık
                cv2.LINE_AA)

    # Görüntüyü göster
    cv2.imshow("Image", img)

    # Kullanıcı bir tuşa basana kadar bekle
    cv2.waitKey(0)
    cv2.destroyAllWindows()
