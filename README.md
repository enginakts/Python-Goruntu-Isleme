# Computer Vision Projects with OpenCV

Bu repo, OpenCV kütüphanesi kullanılarak geliştirilmiş çeşitli görüntü işleme ve bilgisayarlı görü projelerini içermektedir.

## 🚀 Projeler

### 1. YOLOv8 Nesne Tespiti
- Kullanıcı dostu arayüz ile nesne tespiti
- Eşik değeri ayarlama özelliği
- Gerçek zamanlı nesne tespiti
- Sonuçların görsel olarak işaretlenmesi

### 2. Poz Tespiti
- YOLOv8-pose modeli ile gerçek zamanlı iskelet tespiti
- FPS göstergesi
- Eklem noktalarının görselleştirilmesi
- Kamera entegrasyonu

### 3. Plaka Tanıma Sistemi
- Otomatik plaka bölgesi tespiti
- OCR ile plaka metni okuma
- Görüntü ön işleme ve iyileştirme
- Tesseract OCR entegrasyonu

### 4. Şerit Takip Sistemi
- Video üzerinde şerit tespiti
- ROI (İlgi Bölgesi) maskeleme
- Hough dönüşümü ile çizgi tespiti
- Şeritlerin görselleştirilmesi

### 5. Metin Okuma (OCR)
- EasyOCR ile metin tespiti
- Çoklu dil desteği
- Tespit edilen metinlerin görselleştirilmesi
- Güven skoru filtreleme

## 🛠️ Kullanılan Teknolojiler
- Python
- OpenCV
- NumPy
- PyQt5
- YOLOv8
- Tesseract OCR
- EasyOCR

## ⚙️ Kurulum

1. Repo'yu klonlayın:   git clone https://github.com/kullaniciadi/repo-adi.git
 
 Gerekli paketleri yükleyin:   pip install -r requirements.txt


3. Tesseract OCR'ı yükleyin (Plaka tanıma için gerekli):
- Windows: [Tesseract-OCR indirme sayfası](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`

## 🚦 Kullanım

Her proje için ayrı çalıştırma talimatları:

### YOLOv8 Nesne Tespiti:  python Posse/arayuz_yolov.py 

### Poz Tespiti:  python Posse/posse.py

### Plaka Tanıma:  python plaka_okuma.py
 

### Şerit Takibi:  python lane_detection.py 

 
## 📝 Notlar
- Kamera erişimi için gerekli izinlerin verildiğinden emin olun
- YOLOv8 modellerinin indirilmiş olması gerekir
- Tesseract OCR'ın doğru şekilde kurulduğundan emin olun

## 🤝 Katkıda Bulunma
1. Bu repo'yu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun

## 📄 Lisans
Bu proje [MIT](LICENSE) lisansı altında lisanslanmıştır.
