import cv2

# Resim yolu
img_path = "Images/img.jpeg"

# Resmi yükle
img = cv2.imread(img_path)

# Resmin başarıyla yüklenip yüklenmediğini kontrol et
if img is None:
    print("Resim yüklenemedi. Lütfen dosya yolunu kontrol edin.")
else:
    # Gri tonlamaya çevir
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Eşikleme işlemi
    _, th = cv2.threshold(img_gray, 120, 230, cv2.THRESH_BINARY)

    # Görüntüleri göster
    cv2.imshow("Original Image", img)
    cv2.imshow("Grayscale Image", img_gray)
    cv2.imshow("Thresholded Image", th)

    # Kullanıcı bir tuşa basana kadar bekle
    cv2.waitKey(0)
    cv2.destroyAllWindows()
