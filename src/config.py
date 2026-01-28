import torch
from pathlib import Path

# Base directory is the project root (2 levels up from src/)
BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG = {
    # PATHS
    "data_dir": BASE_DIR / "data_padded/lines",
    "metafile": BASE_DIR / "data_padded/METAFILE.tsv",
    "alphabet": BASE_DIR / "data_padded/alphabet.txt",
    
    # Updated to fit new folder structure
    "save_path": BASE_DIR / "checkpoints/crnn_resnet18_best_4.pth", 
    "log_dir": BASE_DIR / "logs/runs/crnn_experiment_final",
    "log_file": BASE_DIR / "logs/training_log.txt",
    "csv_file": BASE_DIR / "logs/training_metrics.csv",

    # SETTINGS (Preserved exactly)
    "image_h": 64,
    "image_w": 800,
    "batch_size": 8,        
    "num_epochs": 1,        
    "learning_rate": 0.0001,
    "log_interval": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}