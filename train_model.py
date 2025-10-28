import os
# --- FIX: FORCE CPU USAGE TO AVOID CUDA/GPU ERRORS ---
# Setting CUDA_VISIBLE_DEVICES to '-1' instructs TensorFlow to bypass the GPU entirely.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# ---------------------------------------------------

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import tensorflow as tf

DATASET_DIR = "dataset"
MODEL_PATH = "disease_model.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 8   # increase if you have more time/data

# confirm dataset exists
if not os.path.exists(DATASET_DIR):
    raise SystemExit("Put your images under ./dataset/<label>/")

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Your Convolutional Neural Network (CNN) Model definition
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    # The output layer size is determined by the number of classes found in your dataset folder structure
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# This is the line that was crashing due to the GPU error
print("Starting model training on CPU...")
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

model.save(MODEL_PATH)
print("Saved model to", MODEL_PATH)
# This provides the necessary class-index mapping for prediction later
print("Class mapping:", train_gen.class_indices)
