"""
Downloads both Kaggle datasets into data/ using kagglehub.

    pip install kagglehub
    python scripts/download_data.py

Public datasets usually download without login. If it asks for credentials,
put your kaggle.json in ~/.kaggle/ (Kaggle > Settings > API > Create New Token).
"""
import glob
import os
import shutil

import kagglehub

DATASETS = {
    "data/nse": "vinay20/stock-market-nse-indices-daily-data2000",
    "data/mutual_funds": "ravibarnawal/mutual-funds-india-detailed",
}

for target, slug in DATASETS.items():
    path = kagglehub.dataset_download(slug)
    os.makedirs(target, exist_ok=True)
    for f in glob.glob(os.path.join(path, "**", "*.csv"), recursive=True):
        shutil.copy(f, target)
        print("copied", os.path.basename(f), "->", target)
