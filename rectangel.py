import cv2

img_path = "Images/img.jpeg"

img = cv2.imread(img_path)


cv2.rectangle(img,(150,80),(750,450),(255,0,0),3)

cv2.imshow("img",img)
cv2.waitKey()
cv2.destroyAllWindows()