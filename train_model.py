# train_model.py — robust trainer with helpful debug output
import os
import sys
import traceback
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ----- CONFIG -----
DATASET_DIR = "dataset"
MODEL_PATH = "disease_model.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 8

# ----- HELPERS -----
def list_dataset_folders(base):
    if not os.path.exists(base):
        return []
    return [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]

def count_images(base):
    counts = {}
    for label in list_dataset_folders(base):
        p = os.path.join(base, label)
        files = [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        counts[label] = len(files)
    return counts

# ----- PRECHECKS -----
print("TensorFlow version:", tf.__version__)
print("Checking dataset folder:", DATASET_DIR)

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

if not any(v > 0 for v in counts.values()):
    print("ERROR: No image files (.jpg/.png) found in dataset subfolders.")
    sys.exit(1)

# If some classes are tiny, suggest lowering batch size or using augmentation
min_count = min(counts.values()) if counts else 0
if min_count < 10:
    print("WARNING: One or more classes have very few images (<10).")
    print("  Training may not be stable. Consider adding more images or use transfer learning.")

# ----- DATA GENERATORS -----
try:
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
except Exception as e:
    print("ERROR while creating data generators:")
    traceback.print_exc()
    sys.exit(1)

num_classes = train_gen.num_classes
print("Number of classes:", num_classes)
print("Class indices:", train_gen.class_indices)

# ----- BUILD MODEL -----
try:
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
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
except Exception as e:
    print("ERROR while building/compiling model:")
    traceback.print_exc()
    sys.exit(1)

# ----- TRAIN -----
try:
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
except tf.errors.ResourceExhaustedError as e:
    print("OOM (ResourceExhaustedError): try lowering BATCH_SIZE or IMG_SIZE, or use GPU with more memory.")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print("ERROR during training:")
    traceback.print_exc()
    sys.exit(1)

# ----- SAVE -----
try:
    model.save(MODEL_PATH)
    print("Saved model to", MODEL_PATH)
except Exception as e:
    print("ERROR while saving model:")
    traceback.print_exc()
    sys.exit(1)

print("Training finished successfully.")
print("Class mapping:", train_gen.class_indices)
