import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.utils import class_weight
import numpy as np

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.25
)

train_set = train_datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=2,
    class_mode='binary',
    subset='training'
)

val_set = train_datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=2,
    class_mode='binary',
    subset='validation'
)

# 🧠 Use MobileNetV2 as base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pretrained layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ⚖️ Compute class weights (for imbalance)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_set.classes),
    y=train_set.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# 🚀 Train the model
model.fit(
    train_set,
    validation_data=val_set,
    epochs=8,
    class_weight=class_weights
)

# Save model
model.save('pneumonia_model.h5')
print("✅ Transfer learning model saved successfully as pneumonia_model.h5")
