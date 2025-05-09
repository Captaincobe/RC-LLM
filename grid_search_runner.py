import os
import subprocess
import csv
from itertools import product
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='CICIDS', choices=["TONIoT","DoHBrw", "CICIDS", "CICMalMen"])
args = parser.parse_args()

dataset_name = args.dataset_name
param_grid = {
    'lr': [1e-4, 5e-4, 1e-3, 1e-2],
    'dropout': [0.3, 0.5, 0.7],
    'weight_decay': [5e-4, 1e-3],
    'batch': [16, 32],
    'hid': [64, 128],
    'dc_loss': [0, 0.2, 0.4, 0.8, 1, 1.5, 2]
}

# param_grid = {
#     'lr': [1e-4,],
#     'dropout': [0.3],
#     'weight_decay': [5e-4],
#     'batch': [16],
#     'hid': [128,256],
#     'dc_loss': [2.1, 2.5, 3]
# }

param_names = list(param_grid.keys())
param_combinations = list(product(*param_grid.values()))

results = []
log_dir = f"grid_logs/{dataset_name}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"run-{time.strftime('%Y%m%d-%H%M')}.log")

for i, combo in enumerate(param_combinations):
    param_str = " ".join([f"--{name} {val}" for name, val in zip(param_names, combo)])
    
    command = f"python main.py --dataset_name {dataset_name} {param_str}"
    print(f"[{i+1}/{len(param_combinations)}] Running: {command}")
        
    temp_output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    full_output = temp_output.stdout

    acc = None
    for line in full_output.splitlines():
        if "Val Acc=" in line:
            try:
                acc = float(line.strip().split("Val Acc=")[-1])
            except:
                pass

    with open(log_file, "a") as log:
        log.write(f"Run {i+1}/{len(param_combinations)} | {command}\n")
        log.write(f"Val Acc = {acc}\n\n")

    results.append((combo, acc))

# csv_file = os.path.join(log_dir, "grid_search_results.csv")
# with open(csv_file, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(param_names + ["val_acc"])
#     for combo, acc in results:
#         writer.writerow(list(combo) + [acc])


best_result = max(results, key=lambda x: x[1] if x[1] is not None else 0)
print("✅ Best combination:", dict(zip(param_names, best_result[0])))
print("🎯 Best Val Accuracy:", best_result[1])


# 保存最佳参数组合到文件
best_combo_file = os.path.join(log_dir, "best_params.txt")
with open(best_combo_file, "w") as f:
    f.write("Best parameter combination:\n")
    for name, val in zip(param_names, best_result[0]):
        f.write(f"{name}: {val}\n")
    f.write(f"Val Accuracy: {best_result[1]}\n")


