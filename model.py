import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

def squeeze_excite_block(input_tensor, ratio=16):
    """Squeeze and Excitation bloğu"""
    filters = input_tensor.shape[-1]
    
    # Squeeze operation
    squeeze = layers.GlobalAveragePooling2D()(input_tensor)
    
    # Excitation operation
    excitation = layers.Dense(filters // ratio, activation='relu')(squeeze)
    excitation = layers.Dense(filters, activation='sigmoid')(excitation)
    excitation = layers.Reshape((1, 1, filters))(excitation)
    
    # Scale
    scale = layers.multiply([input_tensor, excitation])
    
    return scale

def create_se_resnet(sinif_sayisi=10, input_shape=(224, 224, 3)):
    """SENet mimarisi (ResNet50 tabanlı)"""
    
    # Base model olarak ResNet50 kullan
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Base model katmanlarını dondur
    for layer in base_model.layers:
        layer.trainable = False
    
    # SE bloklarını ekle
    x = base_model.output
    
    # Her konvolüsyon bloğuna SE bloğu ekle
    for i in range(1, 6):
        if i > 3:  # Son bloklara SE ekle
            block_name = f'conv{i}_block3_out'
            try:
                block_output = base_model.get_layer(block_name).output
                x = squeeze_excite_block(block_output)
            except:
                continue
    
    # Sınıflandırma katmanları
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    predictions = layers.Dense(sinif_sayisi, activation='softmax')(x)
    
    # Model oluştur
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def model_olustur(sinif_sayisi=10):
    """Ana model oluşturma fonksiyonu"""
    model = create_se_resnet(sinif_sayisi=sinif_sayisi)
    
    return model 