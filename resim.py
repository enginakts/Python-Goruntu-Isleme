import cv2
import os

# Görüntü yolu
img_path = "Images/img.jpeg"

# Görüntü yolunun varlığını kontrol et
if not os.path.exists(img_path):
    print(f"Hata: '{img_path}' dosyası bulunamadı!")
    exit()

# Görüntüyü oku
img = cv2.imread(img_path)

# Görüntünün başarıyla okunup okunmadığını kontrol et
if img is None:
    print("Hata: Görüntü dosyası okunamadı!")
    exit()


# Görüntüyü göster
cv2.imshow("img", img)
print("Pencereyi kapatmak için bir tuşa basın...")


print(img.shape)
# Tuşa basmayı bekle ve pencereyi kapat
cv2.waitKey(0)
cv2.destroyAllWindows()
