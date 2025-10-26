# prepare_dataset.py
import os
import random
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

# =========================================================================================
#                            НАСТРОЙКИ: ЗАПУСКАТЬ 1 РАЗ
# =========================================================================================

SOURCE_PHOTO_DIR = Path("photo") 
SOURCE_LABELS_DIR = Path("labels")

OUTPUT_DIR = Path("dataset_yolo")

# 90% на обучение, 10% на валидацию
VAL_SIZE = 0.1

AUG_SUFFIXES = [
    'rotlbrlrotl', 'rotbrlrotl', 'rotlbrlrot', 'brlrotl', 'rotlbrrot', 'rotbrlrot', 
    'rotlbrrotl', 'rotbrrotl', 'brrotl', 'rotlbrl', 'rotbrl', 'brlrot', 'rotlrot', 
    'rotrot', 'brrot', 'rotbr', 'rotl', 'brl', 'rotrotl', 'rotlrotl', 'rrot', 'rott',
    'rot', 'br', 'l', 't', 'flip' 
]

# =========================================================================================

def get_patient_id(filename_stem, suffixes):
    # извлекает ID пациента, удаляя самый длинный подходящий суффикс с конца
    for suffix in suffixes:
        if filename_stem.endswith(suffix):
            return filename_stem[:-len(suffix)]
    return filename_stem

def prepare_dataset():

    AUG_SUFFIXES.sort(key=len, reverse=True)
    patient_files = defaultdict(list)
    image_paths = list(SOURCE_PHOTO_DIR.glob("*.jpeg")) + list(SOURCE_PHOTO_DIR.glob("*.jpg"))

    for img_path in image_paths:
        patient_id = get_patient_id(img_path.stem, AUG_SUFFIXES)
        patient_files[patient_id].append(img_path)

    print(f"Найдено {len(image_paths)} изображений, сгруппированных в {len(patient_files)} уникальных пациентов.")
    print(f"Разделение пациентов на train ({1-VAL_SIZE:.0%}) / val ({VAL_SIZE:.0%}) ...")
    patient_ids = list(patient_files.keys())
    random.seed(42)
    random.shuffle(patient_ids)

    split_index = int(len(patient_ids) * (1 - VAL_SIZE))
    train_patient_ids = set(patient_ids[:split_index])
    val_patient_ids = set(patient_ids[split_index:])
    print(f"Пациентов в train: {len(train_patient_ids)}, в val: {len(val_patient_ids)}")

    print(f"Создание папки '{OUTPUT_DIR}' и копирование файлов...")
    if OUTPUT_DIR.exists():
        # если папка есть, то удалить ее
        shutil.rmtree(OUTPUT_DIR)

    # все необходимые подпапки
    train_img_dir = OUTPUT_DIR / "images" / "train"
    val_img_dir = OUTPUT_DIR / "images" / "val"
    train_labels_dir = OUTPUT_DIR / "labels" / "train"
    val_labels_dir = OUTPUT_DIR / "labels" / "val"
    for p in [train_img_dir, val_img_dir, train_labels_dir, val_labels_dir]:
        p.mkdir(parents=True, exist_ok=True)

    for patient_id, images in patient_files.items():
        if patient_id in train_patient_ids:
            img_dest, label_dest = train_img_dir, train_labels_dir
        else:
            img_dest, label_dest = val_img_dir, val_labels_dir
        
        for img_path in images:
            shutil.copy(img_path, img_dest)
            label_path = SOURCE_LABELS_DIR / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy(label_path, label_dest)

    print("data.yaml...")
    absolute_output_path = os.path.abspath(OUTPUT_DIR)
    data_yaml = {
        'path': absolute_output_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': { 0: 'foreign_body' }
    }
    
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False, default_flow_style=False)

if __name__ == "__main__":
    prepare_dataset()