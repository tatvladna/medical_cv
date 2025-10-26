import logging
import os
import optuna
from ultralytics import YOLO
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
import src.config as config
import subprocess

def log_subprocess_error(error, trial_num=None):
    trial_info = f"в Optuna Trial #{trial_num} " if trial_num is not None else ""
    logging.error(f"ОБУЧЕНИЕ {trial_info}ЗАВЕРШИЛОСЬ С КРИТИЧЕСКОЙ ОШИБКОЙ")
    logging.error(f"Команда, которая не удалась: {' '.join(error.cmd)}")
    
    # повторно запускается команда, чтобы захватить ее вывод
    result = subprocess.run(error.cmd, capture_output=True, text=True, encoding='utf-8')
    
    logging.error("--- СТАНДАРТНЫЙ ВЫВОД (stdout) ПОДПРОЦЕССА ---")
    logging.error(result.stdout if result.stdout else "Вывод stdout пуст.")
    logging.error("--- ВЫВОД ОШИБОК (stderr) ПОДПРОЦЕССА ---")
    logging.error(result.stderr if result.stderr else "Вывод stderr пуст.")

def run_baseline_training():
    logging.info("--- Начинается базовое обучение (проверка) ---")
    model = YOLO(config.MODEL_TYPE)
    try:
        model.train(
            data=config.DATA_YAML_PATH,
            epochs=config.BASELINE_EPOCHS,
            imgsz=config.IMG_SIZE,
            project=config.PROJECT_NAME,
            name='baseline_training',
            device=config.DEVICE,
            batch=config.BATCH_SIZE,
            workers=8
        )
        logging.info("--- Базовое обучение завершено ---")
    except subprocess.CalledProcessError as e:
        log_subprocess_error(e)
        raise e  

def run_hyperparameter_tuning():

    def objective(trial):
        model = YOLO(config.MODEL_TYPE)
        try:
            results = model.train(
                data=config.DATA_YAML_PATH,
                epochs=config.EPOCHS_OPTUNA,
                imgsz=config.IMG_SIZE,
                project=config.PROJECT_NAME,
                name=f'optuna_trial_{trial.number}',
                lr0=trial.suggest_float('lr0', 1e-5, 1e-1, log=True),
                momentum=trial.suggest_float('momentum', 0.8, 0.98),
                weight_decay=trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
                degrees=trial.suggest_float('degrees', 0.0, 45.0),
                translate=trial.suggest_float('translate', 0.0, 0.3),
                scale=trial.suggest_float('scale', 0.1, 0.9),
                fliplr=trial.suggest_float('fliplr', 0.0, 0.5),
                verbose=False, 
                device=config.DEVICE,
                batch=config.BATCH_SIZE,
                workers=8
            )
            return results.seg.map
        except subprocess.CalledProcessError as e:
                log_subprocess_error(e, trial_num=trial.number)


    logging.info("--- Оптимизация гиперпараметров с Optuna ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=config.N_TRIALS)

    logging.info("--- Подбор гиперпараметров завершен ---")
    logging.info(f"Лучший результат (mAP seg): {study.best_value}")
    logging.info(f"Лучшие гиперпараметры: {study.best_params}")

    try:
        os.makedirs(config.OPTUNA_PLOTS_DIR, exist_ok=True)
        plot_optimization_history(study).write_image(os.path.join(config.OPTUNA_PLOTS_DIR, 'history.png'))
        plot_param_importances(study).write_image(os.path.join(config.OPTUNA_PLOTS_DIR, 'importances.png'))
        plot_slice(study).write_image(os.path.join(config.OPTUNA_PLOTS_DIR, 'slice.png'))
        logging.info(f"Графики Optuna сохранены в: {config.OPTUNA_PLOTS_DIR}")
    except Exception as a:
        pass
    
    return study.best_params

def run_final_training(best_params):
    logging.info("--- Начинаем финальное обучение ---")
    model = YOLO(config.MODEL_TYPE)
    try:
        model.train(
            data=config.DATA_YAML_PATH,
            epochs=config.EPOCHS_FINAL,
            imgsz=config.IMG_SIZE,
            project=config.PROJECT_NAME,
            name='final_best_model',
            device=config.DEVICE,
            batch=config.BATCH_SIZE,
            workers=8,
            **best_params
        )
        logging.info(f"--- Финальное обучение завершено. Модель сохранена в: {config.BEST_MODEL_PATH} ---")
    except subprocess.CalledProcessError as e:
        log_subprocess_error(e)