import cv2
import numpy as np

def process_frame(frame):
    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gürültüyü azaltmak için Gaussian Blur uygula
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Kenar tespiti için Canny algoritması uygula - parametreleri güncelle
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # İlgilenilen bölge (ROI) için maske oluştur
    height, width = frame.shape[:2]
    roi_vertices = np.array([
        [(width * 0.1, height),      # Sol alt
         (width * 0.45, height * 0.6),  # Sol üst
         (width * 0.55, height * 0.6),  # Sağ üst
         (width * 0.9, height)]      # Sağ alt
    ], dtype=np.int32)

    # ROI maskeleme
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [roi_vertices], 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough dönüşümü parametrelerini iyileştir
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi/180,
        threshold=40,
        minLineLength=40,
        maxLineGap=100
    )

    line_image = np.zeros_like(frame)
    
    if lines is not None:
        # Sol ve sağ şeritleri ayır
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            
            # Eğim filtreleme - daha sıkı filtrele
            if abs(slope) < 0.4:  # Çok yatay çizgileri filtrele
                continue
            if abs(slope) > 2:    # Çok dik çizgileri filtrele
                continue
                
            # Orta çizgiye göre sol/sağ şerit ayrımı
            if x1 < width * 0.5:
                if slope < 0:
                    left_lines.append(line[0])
            else:
                if slope > 0:
                    right_lines.append(line[0])

        def smooth_line(lines, prev_line=None):
            if not lines:
                return prev_line
                
            # Medyan filtreleme uygula
            x1s, y1s, x2s, y2s = zip(*lines)
            x1_med = int(np.median(x1s))
            y1_med = int(np.median(y1s))
            x2_med = int(np.median(x2s))
            y2_med = int(np.median(y2s))
            
            # Eğer önceki çizgi varsa, yumuşatma uygula
            if prev_line is not None:
                smoothing = 0.8
                x1_med = int(smoothing * prev_line[0] + (1 - smoothing) * x1_med)
                y1_med = int(smoothing * prev_line[1] + (1 - smoothing) * y1_med)
                x2_med = int(smoothing * prev_line[2] + (1 - smoothing) * x2_med)
                y2_med = int(smoothing * prev_line[3] + (1 - smoothing) * y2_med)
                
            return [x1_med, y1_med, x2_med, y2_med]

        # Şeritleri çiz
        if left_lines:
            left_line = smooth_line(left_lines)
            cv2.line(line_image, (left_line[0], left_line[1]), 
                    (left_line[2], left_line[3]), (0, 0, 255), 8)
            
        if right_lines:
            right_line = smooth_line(right_lines)
            cv2.line(line_image, (right_line[0], right_line[1]), 
                    (right_line[2], right_line[3]), (0, 0, 255), 8)

    # Orijinal görüntü ile şerit çizgilerini birleştir
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    
    return result

# Video yakalama
cap = cv2.VideoCapture('Video/yol_serit_video.mp4')

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Kareyi işle
    result = process_frame(frame)
    
    # Sonuçları göster
    cv2.imshow('Şerit Takibi', result)
    
    # 'q' tuşuna basılınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()











