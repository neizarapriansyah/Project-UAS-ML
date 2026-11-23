import random
import shutil
import csv
from pathlib import Path
from collections import Counter

from PIL import Image
from sklearn.model_selection import train_test_split


# ==========================================
# 1. KONFIGURASI
# ==========================================

SOURCE_DIR = Path("dataset/raw/02. Resized Images")
OUTPUT_DIR = Path("dataset/processed")

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

IMAGE_SIZE = (224, 224)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ==========================================
# 2. VALIDASI KONFIGURASI
# ==========================================

if abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) > 1e-9:
    raise ValueError("Total rasio dataset harus sama dengan 1.")

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Folder dataset tidak ditemukan: {SOURCE_DIR}"
    )

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Folder {OUTPUT_DIR} sudah berisi file. "
        "Pindahkan atau kosongkan folder tersebut sebelum menjalankan ulang."
    )


# ==========================================
# 3. MEMBACA DAN MEMVALIDASI DATASET
# ==========================================

dataset = {}
invalid_files = []

print("\nMemulai validasi dataset...\n")

for class_name in CLASSES:

    class_dir = SOURCE_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Folder kelas tidak ditemukan: {class_dir}"
        )

    valid_images = []

    for image_path in sorted(class_dir.iterdir()):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        try:
            with Image.open(image_path) as image:

                image.verify()

            with Image.open(image_path) as image:

                if image.size != IMAGE_SIZE:
                    raise ValueError(
                        f"Ukuran gambar tidak sesuai: {image.size}"
                    )

                if image.mode != "RGB":
                    raise ValueError(
                        f"Mode warna tidak sesuai: {image.mode}"
                    )

            valid_images.append(image_path)

        except Exception as error:

            invalid_files.append({
                "file": str(image_path),
                "error": str(error)
            })

    if len(valid_images) < 2:
        raise ValueError(
            f"Kelas {class_name} memiliki terlalu sedikit gambar valid."
        )

    dataset[class_name] = valid_images

    print(
        f"Kelas {class_name}: {len(valid_images)} gambar valid"
    )

# ==========================================
# 4. MEMBAGI DATASET
# ==========================================

train_data = []
validation_data = []
test_data = []

print("\nMembagi dataset...\n")

for class_index, (class_name, images) in enumerate(dataset.items()):

    # Seed berbeda untuk setiap kelas
    class_seed = RANDOM_STATE + class_index

    train_images, temporary_images = train_test_split(
        images,
        test_size=0.30,
        random_state=class_seed,
        shuffle=True
    )

    validation_images, test_images = train_test_split(
        temporary_images,
        test_size=0.50,
        random_state=class_seed + 100,
        shuffle=True
    )

    train_data.extend(
        [(image, class_name) for image in train_images]
    )

    validation_data.extend(
        [(image, class_name) for image in validation_images]
    )

    test_data.extend(
        [(image, class_name) for image in test_images]
    )


# ==========================================
# 5. MENYIMPAN DATASET
# ==========================================

splits = {
    "train": train_data,
    "validation": validation_data,
    "test": test_data
}

print("Menyimpan gambar ke folder processed...\n")

manifest = []

for split_name, samples in splits.items():

    for image_path, class_name in samples:

        destination_dir = OUTPUT_DIR / split_name / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination_path = destination_dir / image_path.name

        shutil.copy2(image_path, destination_path)

        manifest.append({
            "split": split_name,
            "class": class_name,
            "filename": image_path.name,
            "path": str(destination_path)
        })


# ==========================================
# 6. MENYIMPAN MANIFEST CSV
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

manifest_path = OUTPUT_DIR / "dataset_manifest.csv"

with open(manifest_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["split", "class", "filename", "path"]
    )

    writer.writeheader()
    writer.writerows(manifest)


# ==========================================
# 7. LAPORAN HASIL
# ==========================================

print("\n" + "=" * 50)
print("HASIL PREPROCESSING DATASET")
print("=" * 50)

for split_name, samples in splits.items():

    class_counts = Counter(
        class_name for _, class_name in samples
    )

    print(f"\n{split_name.upper()}")
    print(f"Total gambar: {len(samples)}")

    for class_name in CLASSES:
        print(
            f"  {class_name}: {class_counts[class_name]}"
        )

total_images = sum(len(samples) for samples in splits.values())

print("\n" + "=" * 50)
print(f"TOTAL GAMBAR : {total_images}")
print(f"FILE INVALID : {len(invalid_files)}")
print(f"MANIFEST     : {manifest_path}")
print("=" * 50)

if invalid_files:

    invalid_report = OUTPUT_DIR / "invalid_images.csv"

    with open(
        invalid_report,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["file", "error"]
        )

        writer.writeheader()
        writer.writerows(invalid_files)

    print(f"Laporan gambar invalid: {invalid_report}")

print("\nPreprocessing selesai!")
