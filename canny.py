import cv2

# Görüntü dosyasını yükleme
img_path = "Images/img.jpeg"
img = cv2.imread(img_path)

# Görüntünün başarıyla yüklenip yüklenmediğini kontrol edin
if img is None:
    print("Görüntü yüklenemedi. Lütfen dosya yolunu kontrol edin.")
else:
    # Yeni boyutlar (örnek: 640x480)
    new_width = 640
    new_height = 480

    # Görüntüyü yeniden boyutlandırma
    resized_img = cv2.resize(img, (new_width, new_height))

    # Gri tonlara çevirme
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Kenar tespiti
    edges = cv2.Canny(img_gray, 50, 150)

    # Görüntüleri gösterme
    cv2.imshow("Original Resized Image", resized_img)
    cv2.imshow("Grayscale Image", img_gray)
    cv2.imshow("Edges", edges)

    # Kullanıcı bir tuşa basana kadar bekle
    cv2.waitKey(0)

    # Tüm pencereleri kapatma
    cv2.destroyAllWindows()
