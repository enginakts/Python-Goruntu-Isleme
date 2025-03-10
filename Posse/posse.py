from ultralytics import YOLO
import cv2
import time
import numpy as np

# YOLO modelinin yolu ve yapılandırması
MODEL_PATH = "yolov8n-pose.pt"

def initialize_camera(camera_id=0):
    """Kamera başlatma fonksiyonu"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError("Kamera başlatılamadı!")
    return cap

def process_frame(frame, model):
    """Görüntü işleme ve poz tespiti fonksiyonu"""
    # Model tahminlerini al
    results = model(frame, verbose=False)[0]
    
    # Her tespit edilen kişi için işlem yap
    for result in results:
        keypoints = result.keypoints.xy[0].numpy()
        # Her eklem noktası için işaretleme yap
        for keypoint_index, keypoint in enumerate(keypoints):
            x, y = keypoint
            # Eklem noktasını çember ile işaretle (kırmızı yerine yeşil)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # BGR formatında (0,255,0) yeşil renk
            # Eklem noktasının indeksini yaz (metin rengi de yeşil)
            cv2.putText(frame, str(keypoint_index), 
                       (int(x), int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    return frame

def main():
    """Ana program fonksiyonu"""
    try:
        # Model yükleme
        print("Model yükleniyor...")
        model = YOLO("yolov8n-pose")
        print("Model başarıyla yüklendi!")
        
        # Kamera başlatma
        print("Kamera başlatılıyor...")
        cap = initialize_camera()
        
        # FPS sayacı için değişkenler
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            # Kameradan görüntü al
            ret, frame = cap.read()
            if not ret:
                print("Görüntü alınamadı!")
                break
            
            # Görüntü işleme
            processed_frame = process_frame(frame, model)
            
            # FPS hesaplama
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # FPS'i ekrana yazdır
            cv2.putText(processed_frame, f"FPS: {fps}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # İşlenmiş görüntüyü göster
            cv2.imshow("Poz Tespiti", processed_frame)
            
            # 'q' tuşuna basılırsa döngüyü sonlandır
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        
    finally:
        # Kaynakları serbest bırak
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


