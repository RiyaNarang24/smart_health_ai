import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----- CONFIGURATION -----
DATASET_DIR = "dataset"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "disease_model.h5"

# ----- HELPER FUNCTIONS -----
def list_dataset_folders(base_dir):
    return [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

def count_images(base_dir):
    counts = {}
    for folder in list_dataset_folders(base_dir):
        folder_path = os.path.join(base_dir, folder)
        counts[folder] = len(os.listdir(folder_path))
    return counts

# ----- PRECHECKS -----
print("TensorFlow version:", tf.__version__)
print("Checking dataset folder:", DATASET_DIR)

if not os.path.exists(DATASET_DIR):
    print(f"ERROR: Dataset folder '{DATASET_DIR}' not found.")
    sys.exit(1)

folders = list_dataset_folders(DATASET_DIR)
if not folders:
    print("ERROR: No subfolders found in", DATASET_DIR)
    print("Each class should be in its own folder. Example:")
    print(" dataset/pneumonia/, dataset/normal/")
    sys.exit(1)

counts = count_images(DATASET_DIR)
print("Found folders and image counts:")
for k, v in counts.items():
    print(f"  - {k}: {v} images")

# ----- DATA AUGMENTATION -----
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

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

# ----- CNN MODEL -----
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
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ----- TRAIN MODEL -----
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ----- SAVE MODEL -----
model.save(MODEL_PATH)
print(f"✅ Model saved successfully as {MODEL_PATH}")
