import cv2

img_path = "Images/img.jpeg"

img = cv2.imread(img_path)
merkez = (150,150)
eksenler = (50,30)

aci = 50
baslangic_acisi = 20
bitis_acisi = 350
renk = (0,0,255)
kalinlik = 3

cv2.ellipse(img, merkez,eksenler,aci,baslangic_acisi, bitis_acisi,renk,kalinlik)

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()