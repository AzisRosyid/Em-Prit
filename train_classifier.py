import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# --- Konfigurasi ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = "bird_classifier.h5"

DATASET_BASE_DIR = "bird_species_dataset"
TRAIN_DIR = os.path.join(DATASET_BASE_DIR, "data", "Train")
VALID_DIR = os.path.join(DATASET_BASE_DIR, "data", "Test")


# --- 1. Persiapan dan Augmentasi Data ---
print("Memulai persiapan data dari direktori lokal...")

if not os.path.exists(TRAIN_DIR) or not os.path.exists(VALID_DIR):
    print(f"Error: Direktori '{TRAIN_DIR}' atau '{VALID_DIR}' tidak dapat ditemukan.")
    print(
        "Pastikan Anda sudah mengekstrak file .zip dan struktur foldernya benar (misal: bird_species_dataset/data/Train)."
    )
    exit()

print(f"Folder 'train' ditemukan di: {TRAIN_DIR}")
print(f"Folder 'valid/test' ditemukan di: {VALID_DIR}")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    VALID_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)
print(f"Ditemukan {num_classes} kelas. Contoh: {class_names[:5]}...")

# --- 2. Perancangan dan Pengembangan Model AI (Transfer Learning) ---
print("Membangun model menggunakan MobileNetV2...")

base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)

base_model.trainable = False

model = Sequential(
    [
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# --- 3. Pelatihan Model ---
print("\nMemulai pelatihan model...")

history = model.fit(
    train_generator, epochs=EPOCHS, validation_data=validation_generator
)

# --- 4. Simpan Model yang Telah Dilatih ---
print(f"Pelatihan selesai. Menyimpan model ke '{MODEL_SAVE_PATH}'...")
model.save(MODEL_SAVE_PATH)
print("Model berhasil disimpan.")

# --- 5. Visualisasi Hasil Training (Opsional) ---
acc = history.history["accuracy"]
val_acc = history.history["validation_accuracy"]
loss = history.history["loss"]
val_loss = history.history["validation_loss"]

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.savefig("training_history.png")
plt.show()
