import cv2
import numpy as np
import imutils

# Görüntü dosya yolu
image_path = "Images/apple.png"

# Görüntüyü okuma
img = cv2.imread(image_path)

# Görüntüyü genişlik olarak 600 piksele yeniden boyutlandırma
img = imutils.resize(img, width=600)

# Görüntüyü gri tonlamaya çevirme
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3x3 boyutunda bir kernel matrisi oluşturma
kernel = np.ones((3, 3), np.uint8)

# Açma işlemi: Gürültüyü kaldırmak için kullanılır (erozyon + genişleme)
acilis = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

# Kapanış işlemi: Küçük boşlukları doldurmak için kullanılır (genişleme + eroziyon)
kapanis = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)

# Morfolojik gradyan: Kenarları belirginleştirmek için kullanılır (genişleme - eroziyon)
degrade = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)

# Top Hat: Görüntünün orijinali ile açma işlemi arasındaki farkı gösterir (parlak bölgeleri vurgular)
tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)

# Black Hat: Görüntünün orijinali ile kapanış işlemi arasındaki farkı gösterir (karanlık bölgeleri vurgular)
blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

# Sonuçları ekranda gösterme
cv2.imshow("Original Image", img)
cv2.imshow("Acilis (Opening)", acilis)
cv2.imshow("Kapanis (Closing)", kapanis)
cv2.imshow("Degrade (Gradient)", degrade)
cv2.imshow("Top Hat", tophat)
cv2.imshow("Black Hat", blackhat)

# Kullanıcı bir tuşa basana kadar bekle
cv2.waitKey(0)

# Açılan tüm pencereleri kapatma
cv2.destroyAllWindows()
