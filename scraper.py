import os
import time
import shutil
import numpy as np
from PIL import Image
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler, BaiduImageCrawler
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

import imagehash
import random

class ImageCollector:
    def __init__(self, save_folder="dataset"):
        self.save_folder = save_folder
        self.temp_folder = "temp_images"

        # Farklı dillerde arama terimleri ve daha fazla varyasyon
        self.siniflar = {
            'kedi': [
                'kedi', 'kediler', 'kedi resmi', 'kedi fotoğrafı', 'ev kedisi',
                'cat', 'cats', 'kitten', 'funny cat', 'cute cat', 'domestic cat', 'cat photo',
                'katze', 'katzen', 'katze foto', 'hauskatze',
                'chat', 'chats', 'chat mignon', 'photo de chat',
                'gato', 'gatos', 'gatito', 'neko', 'cat pictures'
            ],
            'kopek': [
                'köpek', 'köpekler', 'kopek resmi', 'köpek fotoğrafı', 'sevimli köpek',
                'dog', 'dogs', 'puppy', 'puppies', 'funny dog', 'cute dog', 'dog photos', 'dog pictures',
                'hund', 'hunde', 'hündchen', 'welpe',
                'chien', 'chiens', 'chiot', 'perro', 'perros', 'inu'
            ],
            'araba': [
                'araba', 'araba resmi', 'araba fotoğrafı', 'otomobil', 'spor araba',
                'car', 'cars', 'automobile', 'sports car', 'luxury car', 'car photo', 'car pictures',
                'auto', 'autos', 'kraftfahrzeug', 'voiture', 'voitures', 'coche', 'coches', 'kuruma'
            ],
            'ev': [
                'ev', 'ev resmi', 'ev fotoğrafı', 'konut', 'villa', 'apartman',
                'house', 'home', 'building', 'residence', 'mansion', 'house photo', 'house pictures',
                'haus', 'häuser', 'wohnung', 'maison', 'casa', 'ie'
            ],
            'agac': [
                'ağaç', 'ağaçlar', 'orman', 'bitki', 'çam ağacı', 'meşe ağacı',
                'tree', 'trees', 'forest', 'pine tree', 'oak tree', 'tree photo', 'tree pictures',
                'baum', 'bäume', 'wald', 'arbre', 'arbres', 'forêt', 'árbol', 'árboles', 'ki'
            ],
            'insan': [
                'insan', 'insanlar', 'yüz', 'portre', 'kişi', 'erkek', 'kadın',
                'person', 'people', 'human', 'human face', 'portrait', 'man', 'woman', 'people photo',
                'menschen', 'person', 'gesicht', 'personne', 'personnes', 'visage', 'personas', 'hito'
            ],
            'kus': [
                'kuş', 'kuşlar', 'kuş resmi', 'yabani kuş', 'papağan', 'kartal',
                'bird', 'birds', 'sparrow', 'parrot', 'eagle', 'wild bird', 'bird photo', 'bird pictures',
                'vogel', 'vögel', 'oiseau', 'oiseaux', 'pájaro', 'pájaros', 'tori'
            ],
            'cicek': [
                'çiçek', 'çiçekler', 'cicek resmi', 'gül', 'lale', 'ayçiçeği',
                'flower', 'flowers', 'rose', 'tulip', 'sunflower', 'flower photo', 'flower pictures',
                'blume', 'blumen', 'fleur', 'fleurs', 'flor', 'flores', 'hana'
            ],
            'telefon': [
                'telefon', 'cep telefonu', 'akıllı telefon', 'mobil telefon',
                'phone', 'mobile phone', 'smartphone', 'cellphone', 'iphone', 'android phone', 'phone photo',
                'handy', 'smartphone deutsch', 'téléphone', 'portable', 'teléfono', 'móvil', 'keitai'
            ],
            'bilgisayar': [
                'bilgisayar', 'laptop', 'masaüstü bilgisayar', 'dizüstü', 'notebook',
                'computer', 'desktop computer', 'laptop pc', 'notebook', 'pc', 'computer photo',
                'ordinateur', 'portable', 'rechner', 'computadora', 'ordenador', 'pasokon'
            ]
        }

        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(self.temp_folder, exist_ok=True)
        
        # ResNet50 modelini belleğe al (tek seferlik yükleme)
        print("ResNet50 modeli yükleniyor...")
        self.model = ResNet50(weights='imagenet')
        
        # Kaydedilmiş resimlerin hash'lerini tutmak için set
        self.saved_hash_set = set()
        
        # Her sınıf için mevcut resim sayısını izle
        self.collected_counts = {sinif: 0 for sinif in self.siniflar.keys()}
        
        # Mevcut resimleri yükle
        self._load_existing_images()

    def _load_existing_images(self):
        """Mevcut resimleri yükle ve hash'lerini hesapla"""
        print("Mevcut resimler kontrol ediliyor...")
        for sinif in self.siniflar.keys():
            sinif_klasoru = os.path.join(self.save_folder, sinif)
            if os.path.exists(sinif_klasoru):
                resimler = [os.path.join(sinif_klasoru, f) for f in os.listdir(sinif_klasoru) if f.endswith(('.jpg', '.jpeg', '.png'))]
                self.collected_counts[sinif] = len(resimler)
                for resim_yolu in resimler:
                    try:
                        img = Image.open(resim_yolu)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        phash = imagehash.phash(img)
                        self.saved_hash_set.add(phash)
                        img.close()
                    except Exception as e:
                        print(f"Mevcut resim hash hatası {resim_yolu}: {e}")
                print(f"{sinif}: {self.collected_counts[sinif]} mevcut resim bulundu.")

    def temizle(self):
        """Geçici klasörü temizle"""
        if os.path.exists(self.temp_folder):
            shutil.rmtree(self.temp_folder)
        os.makedirs(self.temp_folder, exist_ok=True)

    def _check_prediction(self, decoded_predictions, beklenen_sinif):
        """
        Toplu sınıflandırma sonucunda elde edilen tahmin listesinden,
        beklenen sınıfa ait etiketlerin (skor > 0.2) varlığını kontrol eder.
        """
        sinif_map = {
            'kedi': ['cat', 'tabby', 'persian_cat', 'siamese_cat', 'egyptian_cat', 'tiger_cat'],
            'kopek': ['dog', 'golden_retriever', 'labrador', 'german_shepherd', 'beagle', 'husky',
                     'collie', 'poodle', 'terrier', 'spaniel', 'bulldog', 'hound'],
            'araba': ['car', 'sports_car', 'passenger_car', 'automobile', 'cab', 'jeep', 'limousine',
                     'convertible', 'minivan', 'race_car', 'pickup', 'model'],
            'ev': ['house', 'home', 'building', 'residence', 'mansion', 'dwelling', 'palace', 'cottage'],
            'agac': ['tree', 'forest', 'pine_tree', 'oak', 'maple', 'birch', 'palm', 'juniper', 'cypress'],
            'insan': ['person', 'people', 'human', 'face', 'man', 'woman', 'child', 'boy', 'girl', 'portrait'],
            'kus': ['bird', 'parrot', 'eagle', 'sparrow', 'hawk', 'robin', 'cardinal', 'peacock', 'owl',
                   'finch', 'canary', 'chicken', 'pigeon', 'flamingo', 'jay', 'hummingbird'],
            'cicek': ['flower', 'rose', 'tulip', 'sunflower', 'daisy', 'lily', 'orchid', 'iris',
                     'carnation', 'poppy', 'daffodil', 'blossom', 'petal'],
            'telefon': ['phone', 'mobile_phone', 'smartphone', 'cellphone', 'iphone', 'telephone'],
            'bilgisayar': ['computer', 'laptop', 'desktop_computer', 'pc', 'monitor', 'keyboard', 
                          'notebook', 'screen', 'workstation', 'macbook', 'chromebook']
        }
        kabul_edilen = sinif_map.get(beklenen_sinif, [])
        for (_, etiket, skor) in decoded_predictions:
            for kabul in kabul_edilen:
                if kabul.lower() in etiket.lower() and skor > 0.2:  # Eşik değeri düşürdük
                    return True
        return False

    def sinif_resimleri_topla(self, sinif_adi, hedef_sayi):
        print(f"{sinif_adi} sınıfı için resimler toplanıyor... (Mevcut: {self.collected_counts[sinif_adi]}/{hedef_sayi})")
        
        # Eğer hedef sayıya ulaşıldıysa, işlemi atla
        if self.collected_counts[sinif_adi] >= hedef_sayi:
            print(f"{sinif_adi} sınıfı için hedef sayıya zaten ulaşılmış.")
            return
        
        sinif_klasoru = os.path.join(self.save_folder, sinif_adi)
        os.makedirs(sinif_klasoru, exist_ok=True)
        indirilen = self.collected_counts[sinif_adi]
        
        # Daha fazla deneme için sayıyı artırdık
        max_deneme = 20
        arama_terimleri = list(self.siniflar[sinif_adi])

        deneme = 0
        motor_secimi = 0  # Kullanılacak crawler motoru (0: Google, 1: Bing, 2: Baidu)
        crawlers = [GoogleImageCrawler, BingImageCrawler, BaiduImageCrawler]
        
        while indirilen < hedef_sayi and deneme < max_deneme:
            # Geçici dizini temizleyelim
            self.temizle()
            
            # Arama terimlerini karıştır
            random.shuffle(arama_terimleri)
            
            # Bu turda kullanılacak terimleri seç (her seferinde farklı terimler kullan)
            batch_size = 5  # Her turda kaç terim kullanılacak
            batch_terms = arama_terimleri[:batch_size]
            
            # Farklı motorlar için ThreadPoolExecutor kullanalım
            with ThreadPoolExecutor(max_workers=min(batch_size, 8)) as executor:
                for arama in batch_terms:
                    if indirilen >= hedef_sayi:
                        break
                    
                    # Farklı motorlar arasında geçiş yap
                    crawler_class = crawlers[motor_secimi % len(crawlers)]
                    motor_secimi += 1
                    
                    # Geçici alt klasör oluştur
                    temp_sub_folder = os.path.join(self.temp_folder, f"{arama.replace(' ', '_')}")
                    os.makedirs(temp_sub_folder, exist_ok=True)
                    
                    # Crawler'ı yapılandır
                    crawler = crawler_class(
                        feeder_threads=2,
                        parser_threads=2,
                        downloader_threads=4,
                        storage={'root_dir': temp_sub_folder}
                    )
                    
                    # Resimleri indir
                    try:
                        executor.submit(
                            crawler.crawl,
                            keyword=f"{arama} photo high quality",
                            max_num=200,  # Daha fazla resim dene
                            min_size=(150, 150),  # Biraz daha büyük resimler için
                            file_idx_offset=deneme * 100
                        )
                    except Exception as e:
                        print(f"Crawler hatası ({arama}): {e}")
                        continue
            
            # Tüm indirme işlemleri tamamlandıktan sonra devam et
            time.sleep(5)  # Tüm indirmelerin tamamlanması için biraz bekle
            
            # Resimleri işleme
            temp_klasorler = [os.path.join(self.temp_folder, d) for d in os.listdir(self.temp_folder) 
                            if os.path.isdir(os.path.join(self.temp_folder, d))]
            
            for klasor in temp_klasorler:
                if indirilen >= hedef_sayi:
                    break
                    
                temp_dosyalar = [os.path.join(klasor, dosya) for dosya in os.listdir(klasor)
                               if dosya.endswith(('.jpg', '.jpeg', '.png'))]
                
                if not temp_dosyalar:
                    continue
                
                # Resimleri daha küçük batch'lere ayır (bellek için)
                batch_size = 50  # Bellek için daha küçük batch'ler
                for i in range(0, len(temp_dosyalar), batch_size):
                    if indirilen >= hedef_sayi:
                        break
                        
                    batch_files = temp_dosyalar[i:i+batch_size]
                    pil_resimler = []
                    resimler_hash = []
                    dosya_yolları = []
                    
                    for dosya_yolu in batch_files:
                        try:
                            img = Image.open(dosya_yolu)
                            if img.size[0] < 150 or img.size[1] < 150:
                                img.close()
                                continue
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            phash = imagehash.phash(img)
                            if phash in self.saved_hash_set:
                                img.close()
                                continue
                            pil_resimler.append(img)
                            resimler_hash.append(phash)
                            dosya_yolları.append(dosya_yolu)
                        except Exception as e:
                            print(f"Resim yüklemede hata: {e}")
                            continue
                    
                    if not pil_resimler:
                        continue
                    
                    # Batch işleme
                    batch_imgs = [img.resize((224, 224)) for img in pil_resimler]
                    batch_array = np.array([img_to_array(i) for i in batch_imgs])
                    batch_array = preprocess_input(batch_array)
                    
                    try:
                        tahminler = self.model.predict(batch_array, verbose=0)
                        decoded_list = decode_predictions(tahminler, top=5)
                    except Exception as e:
                        print(f"Batch tahmin hatası: {e}")
                        for img in pil_resimler:
                            try:
                                img.close()
                            except:
                                pass
                        continue
                    
                    # Sınıf kontrolü ve kaydetme
                    for i, decoded in enumerate(decoded_list):
                        if indirilen >= hedef_sayi:
                            break
                        if not self._check_prediction(decoded, sinif_adi):
                            continue
                        
                        hedef_dosya_yolu = os.path.join(sinif_klasoru, f"{sinif_adi}_{indirilen:04d}.jpg")
                        try:
                            pil_resimler[i].save(hedef_dosya_yolu, 'JPEG', quality=85)
                            indirilen += 1
                            self.collected_counts[sinif_adi] = indirilen
                            self.saved_hash_set.add(resimler_hash[i])
                            if indirilen % 10 == 0:
                                print(f"{sinif_adi}: {indirilen}/{hedef_sayi} resim toplandı ({indirilen/hedef_sayi*100:.1f}%)")
                        except Exception as e:
                            print(f"Resim kaydetme hatası: {e}")
                            continue
                    
                    # Resimleri kapat
                    for img in pil_resimler:
                        try:
                            img.close()
                        except:
                            pass
            
            deneme += 1
            # Hızla çok fazla istek göndermemek için kısa bekle
            time.sleep(2)

        print(f"{sinif_adi} sınıfı için toplam {indirilen}/{hedef_sayi} resim toplandı.")
        
    def tum_siniflari_topla(self, hedef_sayi=500):
        siniflar = list(self.siniflar.keys())
        # Sınıfların sırasını karıştır (daha dengeli toplama için)
        random.shuffle(siniflar)
        
        for sinif in siniflar:
            self.sinif_resimleri_topla(sinif, hedef_sayi)
        self.temizle()

def veri_topla(hedef_sayi=500):
    collector = ImageCollector(save_folder="dataset")
    try:
        collector.tum_siniflari_topla(hedef_sayi=hedef_sayi)
    finally:
        collector.temizle()

if __name__ == "__main__":
    veri_topla(hedef_sayi=500)  # Her sınıf için 500 resim hedefi