# 🕸️ Uçtan Uca Veri Bilimi Projesi: Web Kazıma, Analiz ve Tahmin Modeli


### 📌 Giriş ve Proje Özeti

Bu proje, bir web sitesinden dinamik veri toplamayı, toplanan ham veriyi temizlemeyi ve analiz için hazırlamayı ve son olarak bu verilerle eğitilmiş bir Makine Öğrenmesi (ML) modeli oluşturmayı amaçlayan tam teşekküllü bir veri bilimi çalışmasıdır.

Proje, veri toplama (Web Scraping) aşamasından tahmin modelinin oluşturulmasına kadar olan tüm aşamaları modüler Python betikleriyle kapsamaktadır. Bu sayede, karmaşık veri bilimi süreçlerinin adım adım nasıl yönetildiğini gösteren pratik bir referans sunulmaktadır.

### ✨ Temel Bileşenler ve Özellikler

Proje, üç ana modül etrafında yapılandırılmıştır:

1.  **scraper.py (Veri Toplama):** Belirlenen kaynaktan (URL) yapılandırılmış verileri (fiyatlar, özellikler, metinler vb.) otomatik olarak çeken, robust bir web kazıma betiğidir.
2.  **veri_isleme.py (Veri Mühendisliği):** Kazınan ham veriyi yükler, eksik değerleri temizler, aykırı değerleri yönetir, kategorik değişkenleri kodlar ve modeli eğitmek için veriyi ölçeklendirir veya normalize eder.
3.  **model.py (Makine Öğrenmesi):** İşlenmiş veriyi kullanarak tahmin (regresyon) veya sınıflandırma görevini gerçekleştiren bir ML modelini eğitir, performansını değerlendirir ve modeli kaydeder.

### 🛠️ Kullanılan Teknolojiler

Bu projenin temelini oluşturan Python kütüphaneleri:

* **Web Kazıma:** `requests`, `BeautifulSoup` (varsayılmıştır) veya `Scrapy`.
* **Veri İşleme:** `pandas`, `numpy`.
* **Makine Öğrenmesi:** `scikit-learn` veya `tensorflow`/`keras`.
* **Diller:** Python 3.9

### 🚀 Projeyi Çalıştırma

Projeyi yerel makinenizde adım adım çalıştırmak için aşağıdaki talimatları izleyin:

#### 1. Sanal Ortam Kurulumu

Projeyi izole etmek için bir sanal ortam oluşturmanız ve etkinleştirmeniz önerilir:

```bash
python -m venv env
source env/bin/activate  # Linux/macOS
# env\Scripts\activate   # Windows
```

### Bağımlılıkları Yükleme

pip install -r requirements.txt

### Çalıştırma Akışı

1. Veri Toplama: Web kazıma işlemini başlatarak ham veriyi elde edin.

python scraper.py

2. Veri İşleme: Ham veriyi temizleyin ve ML modeline hazır hale getirin.

python veri_isleme.py

3. Model Eğitimi: İşlenmiş veri ile ML modelini eğitin ve kaydedin.

python model.py
