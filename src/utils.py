import logging
import torch
import os

def setup_logging(log_file):
    """Sets up logging to both file and console."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. File Handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 2. Stream Handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def decode_prediction(pred_log_probs, idx2char):
    pred_indices = torch.argmax(pred_log_probs, dim=-1).cpu().numpy()
    decoded_text = []
    prev_char_idx = -1
    for char_idx in pred_indices:
        if char_idx != 0 and char_idx != prev_char_idx:
            decoded_text.append(idx2char[char_idx])
        prev_char_idx = char_idx
    return "".join(decoded_text)

def decode_target(target_indices, idx2char):
    return "".join([idx2char[idx.item()] for idx in target_indices])