import cv2
import typing
from ultralytics import YOLO
import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Nesne Tespit Arayüzü")
        self.setGeometry(100, 100, 1200, 800)
        try:
            self.model = YOLO("yolov8n.pt")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Model yüklenemedi: {str(e)}")
        
        self.threshold = 0.5
        self.image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        self.video_label = QLabel(self)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        
        self.select_image_button = QPushButton("Resim Seç", self)
        self.select_image_button.clicked.connect(self.load_img)
        self.select_image_button.setStyleSheet("padding: 8px;")
        
        self.process_button = QPushButton("Nesne Tespiti Yap", self)
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setStyleSheet("padding: 8px;")
        
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Eşik Değeri:", self)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(int(self.threshold * 100))
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(threshold_layout)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_image_button)
        button_layout.addWidget(self.process_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

    def load_img(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Resimler (*.jpg *.jpeg *.png);;Tüm Dosyalar (*)", options=options)
        if file_path:
            try:
                self.image = cv2.imread(file_path)
                if self.image is None:
                    raise Exception("Resim yüklenemedi")
                self.display_image()
                self.process_button.setEnabled(True)
            except Exception as e:
                QMessageBox.warning(self, "Hata", f"Resim yüklenirken hata oluştu: {str(e)}")

    def display_image(self):
        if self.image is None:
            return
            
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def update_threshold(self, value):
        self.threshold = value / 100.0

    def process_image(self):
        if self.image is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir resim seçin!")
            return
            
        try:
            results = self.model(self.image)[0]
            processed_image = self.image.copy()
            
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                
                if score > self.threshold:
                    cv2.rectangle(
                        processed_image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )
                    
                    label = f"{results.names[int(class_id)]}: {score:.2f}"
                    cv2.putText(
                        processed_image,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
            
            self.image = processed_image
            self.display_image()
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Nesne tespiti sırasında hata oluştu: {str(e)}")
        
            

# Ana uygulama başlatma kodu
if __name__ == '__main__':
    try:
        # QApplication oluştur
        app = QApplication(sys.argv)
        
        # Ana pencereyi oluştur
        window = MainWindow()
        
        # Pencereyi göster
        window.show()
        
        # Uygulamayı çalıştır
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Uygulama başlatılırken hata oluştu: {str(e)}")
    
            

    