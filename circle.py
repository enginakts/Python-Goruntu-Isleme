import cv2
import imutils

# Resim yolu
img_path = "Images/img.jpeg"

# Resmi yükle
img = cv2.imread(img_path)

# Resmin başarıyla yüklenip yüklenmediğini kontrol et
if img is None:
    print("Resim yüklenemedi. Lütfen dosya yolunu kontrol edin.")
else:
    # Resmi yeniden boyutlandır
    resized_img = imutils.resize(img, width=600)
    cv2.circle(resized_img, (20,50),6,(0,0,255),2)
    # Görüntüleri göster
    # cv2.imshow("Original Image", img)
    cv2.imshow("Resized Image", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
