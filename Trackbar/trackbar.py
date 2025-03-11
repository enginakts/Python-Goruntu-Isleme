import cv2

def nothing(x):
    """Trackbar için boş bir callback fonksiyonu"""
    pass

def main():
    # Kamerayı başlat
    cap = cv2.VideoCapture(0)

    # Trackbar penceresini oluştur
    cv2.namedWindow("Trackbar", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold", "Trackbar", 0, 255, nothing)

    while True:
        ret, frame = cap.read()  # Kameradan görüntü al

        if not ret:
            print("Kamera bağlantısı sağlanamadı!")
            break

        # Görüntüyü gri tonlamaya çevir
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Trackbar’dan eşik değeri al
        threshold_value = cv2.getTrackbarPos("Threshold", "Trackbar")

        # Eşikleme işlemi uygula
        _, threshold_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)

        # Görüntüleri göster
        cv2.imshow("Thresholded Frame", threshold_frame)
        cv2.imshow("Original Frame", frame)

        # 'q' tuşuna basıldığında çıkış yap
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
