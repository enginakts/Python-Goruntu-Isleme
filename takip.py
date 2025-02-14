import cv2
import time
import numpy as np
from datetime import datetime

# Sabit değişkenler ve konfigürasyon
CONFIG = {
    'MAX_HISTORY_POINTS': 30,
    'SMOOTHING_WINDOW': 5,
    'MAX_DISPLAY_HEIGHT': 800,
    'INFO_OPACITY': 0.7,
    'CONTROL_PANEL_OPACITY': 0.6,
    'GAUSSIAN_BLUR_KERNEL': (5, 5),
    'TARGET_MARKER_SIZE': 20,
    'RECORDING_FPS': 20,
    'MAX_WIDTH': 1280,  # Maksimum genişlik
    'MAX_HEIGHT': 720,  # Maksimum yükseklik
    'SCALE_FACTOR': 0.75  # Ölçekleme faktörü (0-1 arası)
}

# Kullanılabilir algoritmalar
AVAILABLE_ALGORITHMS = {
    "kcf": {"create": cv2.legacy.TrackerKCF_create, "name": "KCF (Kernelized Correlation Filters)"},
    "csrt": {"create": cv2.legacy.TrackerCSRT_create, "name": "CSRT (Discriminative Correlation Filter)"},
    "mil": {"create": cv2.legacy.TrackerMIL_create, "name": "MIL (Multiple Instance Learning)"},
    "mosse": {"create": cv2.legacy.TrackerMOSSE_create, "name": "MOSSE (Minimum Output Sum of Squared Error)"}
}

class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        # Süreç gürültüsünü optimize et
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32) * 0.05
        # Ölçüm gürültüsünü azalt
        self.kalman.measurementNoiseCov = np.array([[1, 0],
                                                   [0, 1]], np.float32) * 0.005
        self.initialized = False
        self.lost_frames = 0
        self.max_lost_frames = 60  # Tahmin süresini artır
        self.last_prediction = None
        self.velocity = [0, 0]
        self.acceleration = [0, 0]  # İvme takibi ekle
        self.prediction_history = []  # Tahmin geçmişi
        self.confidence = 1.0

    def update(self, point, detected=True):
        if detected:
            measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
            
            if not self.initialized:
                self.kalman.statePre = np.array([[np.float32(point[0])],
                                               [np.float32(point[1])],
                                               [0], [0]], np.float32)
                self.kalman.statePost = np.array([[np.float32(point[0])],
                                                [np.float32(point[1])],
                                                [0], [0]], np.float32)
                self.initialized = True
                self.lost_frames = 0
                self.last_prediction = point
                self.prediction_history = [point]
                self.confidence = 1.0
            else:
                # Hız ve ivme hesapla
                if self.last_prediction is not None:
                    new_velocity = [
                        point[0] - self.last_prediction[0],
                        point[1] - self.last_prediction[1]
                    ]
                    # İvme hesapla
                    self.acceleration = [
                        new_velocity[0] - self.velocity[0],
                        new_velocity[1] - self.velocity[1]
                    ]
                    self.velocity = new_velocity
            
            prediction = self.kalman.predict()
            estimated = self.kalman.correct(measurement)
            
            self.last_prediction = (int(estimated[0][0]), int(estimated[1][0]))
            self.prediction_history.append(self.last_prediction)
            if len(self.prediction_history) > 10:
                self.prediction_history.pop(0)
            
            self.lost_frames = 0
            self.confidence = 1.0
            return self.last_prediction, self.confidence
        else:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                return None, 0.0
            
            # Gelişmiş tahmin
            prediction = self.kalman.predict()
            
            # Hız ve ivmeyi kullan
            predicted_x = prediction[0][0] + self.velocity[0] + 0.5 * self.acceleration[0]
            predicted_y = prediction[1][0] + self.velocity[1] + 0.5 * self.acceleration[1]
            
            # Tahmin güvenilirliğini hesapla
            self.confidence = max(0.0, 1.0 - (self.lost_frames / self.max_lost_frames))
            
            self.last_prediction = (int(predicted_x), int(predicted_y))
            self.prediction_history.append(self.last_prediction)
            if len(self.prediction_history) > 10:
                self.prediction_history.pop(0)
            
            return self.last_prediction, self.confidence

class TrackingSystem:
    def __init__(self):
        self.kalman = KalmanFilter()
        self.tracking_points = []
        self.is_paused = False
        self.is_recording = False
        self.out = None
        self.fps = 0
        self.frame_counter = 0
        self.start_time = time.time()
        self.last_center = None
        self.show_help = False
        self.tracker = None
        self.current_algorithm = None
        self.bbox = None
        self.confidence_threshold = 0.3  # Güven eşiğini düşür
        self.lost_object_counter = 0
        self.max_lost_frames = 60  # Tahmin süresini artır
        self.search_window_size = 1.5  # Arama penceresi boyutu
        
    def initialize_video(self, video_path):
        """Video kaynağını başlatır"""
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise ValueError("Video açılamadı!")
        return self.video.read()

    def select_algorithm(self):
        """Kullanıcıya algoritma seçimi sunar"""
        print("\nKullanılabilir Takip Algoritmaları:")
        for i, (key, value) in enumerate(AVAILABLE_ALGORITHMS.items(), 1):
            print(f"{i}. {value['name']}")
        
        while True:
            try:
                choice = int(input("\nAlgoritma seçin (1-4): "))
                if 1 <= choice <= len(AVAILABLE_ALGORITHMS):
                    self.current_algorithm = list(AVAILABLE_ALGORITHMS.keys())[choice-1]
                    return True
                print("Geçersiz seçim!")
            except ValueError:
                print("Lütfen bir sayı girin!")

    def create_tracker(self):
        """Seçilen algoritmaya göre tracker oluşturur"""
        if self.current_algorithm not in AVAILABLE_ALGORITHMS:
            raise ValueError("Geçersiz algoritma!")
        self.tracker = AVAILABLE_ALGORITHMS[self.current_algorithm]["create"]()

    def update_fps(self):
        """FPS hesaplaması yapar"""
        self.frame_counter += 1
        if self.frame_counter % 30 == 0:
            end_time = time.time()
            self.fps = 30 / (end_time - self.start_time)
            self.start_time = time.time()

    def process_frame(self, frame):
        """Frame'i işler ve görüntü efektlerini uygular"""
        # Orijinal boyutları al
        height, width = frame.shape[:2]
        
        # Yeni boyutları hesapla
        if width > CONFIG['MAX_WIDTH'] or height > CONFIG['MAX_HEIGHT']:
            aspect_ratio = width / height
            if width > height:
                new_width = CONFIG['MAX_WIDTH']
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = CONFIG['MAX_HEIGHT']
                new_width = int(new_height * aspect_ratio)
        else:
            new_width = int(width * CONFIG['SCALE_FACTOR'])
            new_height = int(height * CONFIG['SCALE_FACTOR'])
        
        # Frame'i yeniden boyutlandır
        frame_resized = cv2.resize(frame, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
        return frame_resized

    def draw_tracking_info(self, frame, bbox, success):
        if success or self.lost_object_counter < self.max_lost_frames:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            center_x = int(bbox[0] + bbox[2]/2)
            center_y = int(bbox[1] + bbox[3]/2)
            
            predicted_center, confidence = self.kalman.update((center_x, center_y), success)
            
            if predicted_center is not None:
                self.tracking_points.append(predicted_center)
                if len(self.tracking_points) > CONFIG['MAX_HISTORY_POINTS']:
                    self.tracking_points.pop(0)
                
                smooth_center = self.moving_average(self.tracking_points)
                if smooth_center:
                    center_x, center_y = smooth_center

                if success:
                    self.draw_target_box(frame, p1, p2)
                    color = (0, 255, 0)
                else:
                    # Tahmin modunda genişletilmiş arama penceresi
                    search_w = int(bbox[2] * self.search_window_size)
                    search_h = int(bbox[3] * self.search_window_size)
                    search_x = max(0, center_x - search_w//2)
                    search_y = max(0, center_y - search_h//2)
                    
                    overlay = frame.copy()
                    cv2.rectangle(overlay, 
                                (search_x, search_y),
                                (search_x + search_w, search_y + search_h),
                                (0, 255, 255), 2)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    color = (0, 255, 255)
                
                self.draw_target_marker(frame, center_x, center_y, color)
                self.draw_motion_trail(frame)
                self.draw_confidence_bar(frame, confidence)
                self.draw_info_panel(frame, bbox, success, confidence)
                
                if not success:
                    self.lost_object_counter += 1
                    cv2.putText(frame, f"Tahmin Modu ({confidence:.2f})", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                self.draw_warning(frame)
                self.lost_object_counter = 0
        else:
            self.draw_warning(frame)
            self.tracking_points.clear()
            self.kalman.initialized = False

    def draw_target_box(self, frame, p1, p2):
        """Hedef kutusunu çizer"""
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        
    def draw_target_marker(self, frame, x, y, color=(0, 255, 0)):
        """Hedef işaretleyiciyi çizer"""
        size = CONFIG['TARGET_MARKER_SIZE']
        cv2.circle(frame, (x, y), 4, color, -1)
        cv2.line(frame, (x - size, y), (x + size, y), color, 1)
        cv2.line(frame, (x, y - size), (x, y + size), color, 1)

    def draw_motion_trail(self, frame):
        """Hareket izini çizer"""
        if len(self.tracking_points) > 2:
            for i in range(1, len(self.tracking_points)):
                cv2.line(frame, self.tracking_points[i-1], self.tracking_points[i],
                        (0, 255, 255), 2)

    def draw_confidence_bar(self, frame, confidence):
        """Güven seviyesi çubuğunu çizer"""
        height, width = frame.shape[:2]
        bar_width = 100
        bar_height = 10
        x = width - bar_width - 10
        y = 30
        
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y), (x + int(bar_width * confidence), y + bar_height),
                     (0, 255 * confidence, 255 * (1 - confidence)), -1)
        cv2.putText(frame, f"Güven: {confidence:.2f}", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_info_panel(self, frame, bbox, success, confidence):
        """Bilgi panelini çizer"""
        height, width = frame.shape[:2]
        info_text = [
            f"Algoritma: {self.current_algorithm.upper()}",
            f"FPS: {self.fps:.1f}",
            f"Konum: ({int(bbox[0])}, {int(bbox[1])})",
            f"Boyut: {int(bbox[2])}x{int(bbox[3])}px",
            f"Güven: {confidence:.2f}"
        ]
        
        # Yarı saydam arka plan
        info_bg = frame.copy()
        cv2.rectangle(info_bg, (5, 5), (250, 140), (0, 0, 0), -1)
        cv2.addWeighted(info_bg, CONFIG['INFO_OPACITY'], frame, 
                       1 - CONFIG['INFO_OPACITY'], 0, frame)
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + (i * 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_warning(self, frame):
        """Uyarı mesajını çizer"""
        warning_bg = frame.copy()
        cv2.rectangle(warning_bg, (5, 5), (200, 50), (0, 0, 255), -1)
        cv2.addWeighted(warning_bg, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, "Takip Kaybedildi!", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_control_panel(self, frame):
        """Kontrol panelini çizer"""
        height, width = frame.shape[:2]
        panel_height = 160  # Panel yüksekliğini artırdım
        
        # Yarı saydam panel arka planı
        control_bg = frame.copy()
        cv2.rectangle(control_bg,
                     (10, height - panel_height - 10),  # Sol tarafa taşıdım
                     (300, height - 10),  # Genişliği artırdım
                     (0, 0, 0), -1)
        cv2.addWeighted(control_bg, CONFIG['CONTROL_PANEL_OPACITY'], frame,
                       1 - CONFIG['CONTROL_PANEL_OPACITY'], 0, frame)

        # Kontrol bilgileri - Daha detaylı açıklamalar
        controls = [
            ("KONTROL PANELİ", (0, 255, 255)),  # Başlık - Sarı
            ("Q: Programdan Çıkış", (255, 255, 255)),
            ("R: Yeni Nesne Seçimi", (255, 255, 255)),
            ("S: Video Kaydı Başlat/Durdur", (255, 255, 255)),
            ("P: Videoyu Duraklat/Devam Et", (255, 255, 255)),
            ("H: Yardım Menüsü", (255, 255, 255))
        ]
        
        # Aktif durum göstergeleri
        status = []
        if self.is_paused:
            status.append(("DURAKLATILDI", (0, 0, 255)))  # Kırmızı
        if self.is_recording:
            status.append(("KAYIT YAPILIYOR", (0, 0, 255)))  # Kırmızı
        
        # Kontrolleri çiz
        for i, (text, color) in enumerate(controls):
            y_pos = height - panel_height + (i * 25) + 30
            if i == 0:  # Başlık için özel format
                cv2.putText(frame, text, (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:  # Normal kontroller
                # Tuş kısmını vurgula
                key = text.split(':')[0]
                description = text.split(':')[1]
                
                # Tuş arka planı
                key_bg = frame.copy()
                text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(key_bg, 
                            (20, y_pos - 20),
                            (20 + text_size[0] + 10, y_pos + 5),
                            (50, 50, 50), -1)
                cv2.addWeighted(key_bg, 0.5, frame, 0.5, 0, frame)
                
                # Tuş ve açıklama
                cv2.putText(frame, key, (25, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, description, (25 + text_size[0] + 15, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Durum göstergelerini çiz
        for i, (text, color) in enumerate(status):
            y_pos = height - panel_height + 30
            cv2.putText(frame, text, (width - 200, y_pos + (i * 30)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def draw_active_controls(self, frame):
        """Aktif tuş bilgilerini çizer"""
        height, width = frame.shape[:2]
        
        if self.show_help:
            help_bg = frame.copy()
            cv2.rectangle(help_bg, (width//2 - 200, height//2 - 100),
                         (width//2 + 200, height//2 + 100), (0, 0, 0), -1)
            cv2.addWeighted(help_bg, 0.8, frame, 0.2, 0, frame)
            
            help_text = [
                "YARDIM MENÜSÜ",
                "-------------",
                "Q: Programdan çık",
                "R: Yeni nesne seç",
                "S: Kayıt başlat/durdur",
                "P: Duraklat/devam et",
                "H: Bu menüyü kapat"
            ]
            
            for i, text in enumerate(help_text):
                y_pos = height//2 - 70 + (i * 25)
                if i == 0:  # Başlık
                    cv2.putText(frame, text, (width//2 - 100, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, text, (width//2 - 150, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def handle_recording(self, frame):
        """Video kaydını yönetir"""
        if self.is_recording and self.out is not None:
            height, width = frame.shape[:2]
            cv2.circle(frame, (width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (width - 80, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.out.write(frame)

    def start_recording(self, frame):
        """Video kaydını başlatır"""
        if not self.is_recording:
            filename = f'kayit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(filename, fourcc, CONFIG['RECORDING_FPS'],
                                     (frame.shape[1], frame.shape[0]))
            self.is_recording = True
            print(f"Kayıt başlatıldı: {filename}")
            return True
        return False

    def stop_recording(self):
        """Video kaydını durdurur"""
        if self.is_recording:
            self.out.release()
            self.is_recording = False
            self.out = None
            print("Kayıt durduruldu")
            return True
        return False

    @staticmethod
    def moving_average(points, window_size=CONFIG['SMOOTHING_WINDOW']):
        """Hareketli ortalama hesaplar"""
        if len(points) < window_size:
            return points[-1] if points else None
        return tuple(int(sum(coord[i] for coord in points[-window_size:])/window_size)
                    for i in range(2))

    def cleanup(self):
        """Kaynakları temizler"""
        if self.out is not None:
            self.out.release()
        self.video.release()
        cv2.destroyAllWindows()

def main():
    tracking_system = TrackingSystem()
    scale_factor = None  # Ölçekleme faktörünü saklamak için
    
    try:
        # Video kaynağını başlat
        ret, frame = tracking_system.initialize_video("Video/konya_trafic_camera2.mp4")
        if not ret:
            raise ValueError("İlk frame okunamadı!")

        # İlk frame'in boyutlarını al
        original_height, original_width = frame.shape[:2]

        # Algoritma seçimi
        if not tracking_system.select_algorithm():
            raise ValueError("Algoritma seçilemedi!")

        # Frame'i işle ve ölçekleme oranını hesapla
        frame_display = tracking_system.process_frame(frame)
        display_height, display_width = frame_display.shape[:2]
        
        # Ölçekleme faktörlerini hesapla
        scale_x = display_width / original_width
        scale_y = display_height / original_height

        # İlk nesne seçimi
        print("Takip edilecek nesneyi seçin ve SPACE veya ENTER tuşuna basın")
        bbox = cv2.selectROI("Takip", frame_display, False)
        
        # Tracker'ı başlat
        tracking_system.create_tracker()
        tracking_system.tracker.init(frame_display, bbox)

        while True:
            if tracking_system.is_paused:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('p'):
                    tracking_system.is_paused = False
                    print("Video devam ediyor")
                continue

            ret, frame = tracking_system.video.read()
            if not ret:
                print("Video bitti")
                break

            # Frame'i işle
            frame_display = tracking_system.process_frame(frame)

            # Takip güncelleme
            success, bbox = tracking_system.tracker.update(frame_display)

            # Tuş kontrollerini işle
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Program kapatılıyor...")
                break
            elif key == ord('r'):
                print("Yeni nesne seçin")
                cv2.destroyAllWindows()
                cv2.imshow("Takip", frame_display)
                bbox = cv2.selectROI("Takip", frame_display, False)
                
                if bbox[2] > 0 and bbox[3] > 0:
                    # Yeni tracker oluştur ve işlenmiş frame ile başlat
                    tracking_system.create_tracker()
                    success = tracking_system.tracker.init(frame_display, bbox)
                    
                    if success:
                        tracking_system.tracking_points.clear()
                        tracking_system.kalman.initialized = False
                        tracking_system.last_center = None
                        print("Yeni nesne takibi başlatıldı")
                    else:
                        print("Tracker başlatılamadı, lütfen tekrar deneyin")
                else:
                    print("Geçerli bir seçim yapılmadı")
                continue

            # Takip bilgilerini çiz
            tracking_system.draw_tracking_info(frame_display, bbox, success)

            # Kontrol panelini çiz
            tracking_system.draw_control_panel(frame_display)
            tracking_system.draw_active_controls(frame_display)

            # Kayıt durumunu yönet
            tracking_system.handle_recording(frame_display)

            cv2.imshow("Takip", frame_display)

    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
    finally:
        tracking_system.cleanup()

if __name__ == "__main__":
    main()