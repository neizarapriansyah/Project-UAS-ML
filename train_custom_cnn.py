import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. KONFIGURASI
# ==========================================

DATASET_DIR = Path("dataset/processed")

MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. MEMUAT DATASET
# ==========================================

print("\nMemuat dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "validation",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

# Pastikan urutan kelas konsisten
assert train_ds.class_names == validation_ds.class_names, \
    "Urutan kelas train dan validation berbeda!"

assert train_ds.class_names == test_ds.class_names, \
    "Urutan kelas train dan test berbeda!"

assert num_classes == 26, \
    f"Jumlah kelas seharusnya 26, ditemukan {num_classes}"

print("\nMapping kelas konsisten!")

print("\nDaftar kelas:", class_names)
print("Jumlah kelas:", num_classes)


# Simpan mapping kelas
with open(MODEL_DIR / "class_names.json", "w") as file:
    json.dump(class_names, file, indent=4)


# Optimasi input pipeline
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
validation_ds = validation_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ==========================================
# PEMERIKSAAN INPUT DATA
# ==========================================

images, labels = next(iter(train_ds))

print("\nPemeriksaan batch pertama:")
print("Shape gambar :", images.shape)
print("Tipe data    :", images.dtype)
print("Pixel minimum:", tf.reduce_min(images).numpy())
print("Pixel maksimum:", tf.reduce_max(images).numpy())
print("Label minimum:", tf.reduce_min(labels).numpy())
print("Label maksimum:", tf.reduce_max(labels).numpy())

assert images.shape[1:] == (224, 224, 3)
assert tf.reduce_min(images).numpy() >= 0
assert tf.reduce_max(images).numpy() <= 255
assert tf.reduce_min(labels).numpy() >= 0
assert tf.reduce_max(labels).numpy() < 26

print("Pemeriksaan input berhasil!")


# ==========================================
# 3. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.04),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    )
], name="data_augmentation")


# ==========================================
# 4. MEMBANGUN MODEL CUSTOM CNN
# ==========================================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

x = tf.keras.layers.Conv2D(
    32, 3, activation="relu", padding="same"
)(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(
    64, 3, activation="relu", padding="same"
)(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(
    128, 3, activation="relu", padding="same"
)(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="IsyaraAI_CustomCNN"
)


# ==========================================
# 5. COMPILATION
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0003
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nArsitektur Model:")
model.summary()


# ==========================================
# 6. CALLBACKS
# ==========================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_custom_cnn.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]


# ==========================================
# 7. TRAINING
# ==========================================

print("\nMemulai training Custom CNN...\n")

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ==========================================
# 8. MEMUAT DAN MENYIMPAN MODEL TERBAIK
# ==========================================

best_model = tf.keras.models.load_model(
    MODEL_DIR / "best_custom_cnn.keras"
)

best_model.save(
    MODEL_DIR / "custom_cnn_final.keras"
)

print("\nModel terbaik berhasil disimpan!")


# ==========================================
# 9. VISUALISASI TRAINING
# ==========================================

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Custom CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()


# Loss
plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Custom CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "custom_cnn_training.png"
)

plt.show()


# ==========================================
# 10. EVALUASI TEST DATASET
# ==========================================

print("\nEvaluasi menggunakan test dataset...\n")

test_loss, test_accuracy = best_model.evaluate(test_ds)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ==========================================
# 11. CLASSIFICATION REPORT
# ==========================================

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])

y_prob = best_model.predict(test_ds)

y_pred = np.argmax(y_prob, axis=1)

print("\nDistribusi Prediksi Model:")

predicted_classes, predicted_counts = np.unique(
    y_pred,
    return_counts=True
)

for class_index, count in zip(
    predicted_classes,
    predicted_counts
):
    print(
        f"{class_names[class_index]}: {count} gambar"
    )

print("\nDistribusi Label Asli:")

actual_classes, actual_counts = np.unique(
    y_true,
    return_counts=True
)

for class_index, count in zip(
    actual_classes,
    actual_counts
):
    print(
        f"{class_names[class_index]}: {count} gambar"
    )

report = classification_report(
    y_true,
    y_pred,
    labels=range(num_classes),
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open(
    RESULT_DIR / "classification_report.txt",
    "w"
) as file:
    file.write(report)


# ==========================================
# 12. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(num_classes)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(14, 14))

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    colorbar=False
)

plt.title("Custom CNN Confusion Matrix")
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "custom_cnn_confusion_matrix.png"
)

plt.show()


print("\n" + "=" * 50)
print("TRAINING DAN EVALUASI SELESAI")
print("=" * 50)
