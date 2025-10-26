import logging
from src.utils import setup_logger
from src.train_model import run_baseline_training, run_hyperparameter_tuning, run_final_training
from src.val_eval import evaluate_valset

def main():
    setup_logger("main_run")
    logging.info("======================= START =========================")

    run_baseline_training()
    best_params = run_hyperparameter_tuning()
    run_final_training(best_params)
    evaluate_valset()

    logging.info("====================== FINISH ========================")

if __name__ == "__main__":
    main()


