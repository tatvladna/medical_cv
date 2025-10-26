# evaluation.py
import logging
import os
import numpy as np
from pathlib import Path
import yaml
from ultralytics import YOLO
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Импортируем нашу конфигурацию
import src.config as config

def evaluate_valset():
    logging.info("--------- Оценка модели на ВАЛИДАЦИОННОМ наборе ---------")

    model = YOLO(config.BEST_MODEL_PATH)
    
    model_size_mb = os.path.getsize(config.BEST_MODEL_PATH) / (1024 * 1024)
    logging.info(f"Размер модели: {model_size_mb:.2f} MB")

    with open(config.DATA_YAML_PATH, 'r') as f:
        data_config = yaml.safe_load(f)
    val_img_dir = Path(data_config['path']) / data_config['val']
    val_image_paths = list(val_img_dir.glob("*.jpeg")) + list(val_img_dir.glob("*.jpg"))

    y_true, y_scores = [], []
    logging.info("Расчет ROC-AUC...")
    for img_path in val_image_paths:
        label_path = Path(str(img_path).replace('images', 'labels').replace('.jpeg', '.txt').replace('.jpg', '.txt'))
        y_true.append(1 if label_path.exists() and os.path.getsize(label_path) > 0 else 0)
        results = model.predict(img_path, verbose=False)
        y_scores.append(np.max(results[0].boxes.conf.cpu().numpy()) if results[0].boxes else 0.0)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    logging.info(f"Метрика ROC AUC на ВАЛИДАЦИИ: {roc_auc:.4f}")