import os
import torch
from dotenv import load_dotenv
load_dotenv()
DATA_YAML_PATH = os.getenv('DATA_YAML_PATH')
PROJECT_NAME = os.getenv('PROJECT_NAME')
OPTUNA_PLOTS_DIR = os.getenv('OPTUNA_PLOTS_DIR')
FINAL_PLOTS_DIR = os.getenv('FINAL_PLOTS_DIR')
MODEL_TYPE = os.getenv('MODEL_TYPE')
IMG_SIZE = int(os.getenv('IMG_SIZE'))
EPOCHS_OPTUNA = int(os.getenv('EPOCHS_OPTUNA', 50))
EPOCHS_FINAL = int(os.getenv('EPOCHS_FINAL', 150))
BASELINE_EPOCHS = int(os.getenv('BASELINE_EPOCHS', 10))
N_TRIALS = int(os.getenv('N_TRIALS', 50))
BEST_MODEL_PATH = os.path.join(PROJECT_NAME, 'final_best_model/weights/best.pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))
BASELINE_EPOCHS = int(os.getenv('BASELINE_EPOCHS', 10))