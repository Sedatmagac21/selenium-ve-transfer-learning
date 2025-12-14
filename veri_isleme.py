import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import model_olustur
import tensorflow as tf

def veri_seti_hazirla(base_dir="dataset", img_size=(224, 224)):
    # Sınıf listesi - sabit sınıfları kullan
    siniflar = ['kedi', 'kopek', 'araba', 'ev', 'agac', 
                'insan', 'kus', 'cicek', 'telefon', 'bilgisayar']
    sinif_sayisi = len(siniflar)
    
    # Veri ve etiketleri tutacak listeler
    X = []
    y = []
    
    # Her sınıf için
    for i, sinif in enumerate(siniflar):
        sinif_dir = os.path.join(base_dir, sinif)
        if not os.path.exists(sinif_dir):
            continue
            
        for img_name in os.listdir(sinif_dir):
            try:
                img_path = os.path.join(sinif_dir, img_name)
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img)
                img_array = img_array / 255.0
                X.append(img_array)
                y.append(i)
            except:
                continue
    
    X = np.array(X)
    y = np.array(y)
    y = to_categorical(y, sinif_sayisi)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, sinif_sayisi

def veri_cesitlendirme():
    veri_yolu = 'veri_seti'
    test_yolu = 'test_veri_seti'
    
    # Klasör kontrolü
    if not os.path.exists(veri_yolu) or not os.listdir(veri_yolu):
        raise ValueError(f"'{veri_yolu}' klasörü boş veya mevcut değil!")
    
    print(f"Bulunan sınıflar: {os.listdir(veri_yolu)}")
    
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.15,
        rescale=1./255
    )
    
    print("Eğitim verisi yükleniyor...")
    train_generator = train_datagen.flow_from_directory(
        veri_yolu,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    print("Doğrulama verisi yükleniyor...")
    validation_generator = train_datagen.flow_from_directory(
        veri_yolu,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    if not os.path.exists(test_yolu):
        print(f"'{test_yolu}' klasörü bulunamadı. Önce veriyi_ayir() fonksiyonunu çalıştırın.")
        return None, None, None
        
    print("Test verisi yükleniyor...")
    test_generator = test_datagen.flow_from_directory(
        test_yolu,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def veriyi_ayir():
    # Her sınıf için test verilerini ayır
    siniflar = ['kedi', 'kopek', 'araba', 'ev', 'agac', 
                'insan', 'kus', 'cicek', 'telefon', 'bilgisayar']
    
    for sinif in siniflar:
        kaynak_klasor = f'veri_seti/{sinif}'
        hedef_klasor = f'test_veri_seti/{sinif}'
        
        if not os.path.exists(hedef_klasor):
            os.makedirs(hedef_klasor)
        
        # Sınıftaki tüm dosyaları listele
        dosyalar = os.listdir(kaynak_klasor)
        # %15'ini test için ayır
        test_dosya_sayisi = int(len(dosyalar) * 0.15)
        test_dosyalari = np.random.choice(dosyalar, test_dosya_sayisi, replace=False)
        
        # Test dosyalarını taşı
        for dosya in test_dosyalari:
            kaynak = os.path.join(kaynak_klasor, dosya)
            hedef = os.path.join(hedef_klasor, dosya)
            os.rename(kaynak, hedef)

if __name__ == "__main__":
    from scraper import veri_topla
    
    # Önce veriyi topla
    print("Veri toplama başlıyor...")
    veri_topla(hedef_sayi=500)
    
    # Sonra veri setini hazırla
    print("\nVeri seti hazırlanıyor...")
    X_train, X_test, y_train, y_test, sinif_sayisi = veri_seti_hazirla()
    
    print("\nVeri seti hazır!")
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")
    print(f"Sınıf sayısı: {sinif_sayisi}")
    
    # Veri çeşitlendirme ve bölme
    train_gen, val_gen, test_gen = veri_cesitlendirme()
    
    # Modeli oluştur
    model = model_olustur()
    
    # Modeli derle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),  # Başlangıç learning rate'i
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model checkpoint ve learning rate scheduling ekle
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # İlk eğitim
    print("İlk eğitim başlıyor...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks
    )
    
    # Fine-tuning
    print("\nFine-tuning başlıyor...")
    
    # Son ResNet bloğunu ve SE bloklarını çöz
    for layer in model.layers[-50:]:  # Son 50 katmanı çöz
        layer.trainable = True
    
    # Daha düşük learning rate ile yeniden derle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tuning eğitimi
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        callbacks=callbacks
    )
    
    # Test et
    print("\nTest sonuçları:")
    test_sonuclari = model.evaluate(test_gen)
    print(f"Test kaybı: {test_sonuclari[0]:.4f}")
    print(f"Test doğruluğu: {test_sonuclari[1]:.4f}")
    
    # Modeli kaydet
    model.save('final_senet_model.h5') 