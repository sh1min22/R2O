import logging
import os
from datetime import datetime

current_time = datetime.now()
time_str = current_time.strftime("%Y-%m-%d")
log_filename = f"train_MPC_SF30_{time_str}.log"
# log_filename = f"offline_local_train_{time_str}.log"
# log_filename = f"offline_global_train_{time_str}.log"
# log_filename = f"default_order_{time_str}_1-100.log"
# log_filename = f"mpc_evaluate_{time_str}.log"
# log_filename = f"mpc_evaluate_4.log"
# log_filename = f"evaluate_only_global_{time_str}.log"
# log_filename = f"evaluate_only_local_{time_str}.log"
log_filename = f"Metis_SF30_baseline_{time_str}.log"
# log_filename = f"Metis_SF10_baseline_{time_str}.log"
# log_filename = f"MPC_SF30_rlqpg_{time_str}.log"
log_filename = f"Metis_SF30_rlqpg_{time_str}.log"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_folder = os.path.join(BASE_DIR, 'log')
os.makedirs(log_folder, exist_ok=True)
LOG_FILE = os.path.join(log_folder, log_filename)

logger = logging.getLogger("GlobalLogger")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s")

file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
