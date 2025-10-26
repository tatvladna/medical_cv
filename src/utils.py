import logging
import sys
import os
import optuna
from datetime import datetime

LOG_DIR = "logs"

def setup_logger(file_name):
    os.makedirs(LOG_DIR, exist_ok=True)
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(LOG_DIR, f'{file_name}_{current_time}.log')
    
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    root_logger = logging.getLogger()
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)
        
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    return log_file