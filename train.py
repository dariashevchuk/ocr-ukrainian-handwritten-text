import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import csv

from src.config import CONFIG
from src.utils import setup_logging
from src.dataset import UkrainianOCRDataset, collate_fn
from src.model import OCR_SimpleCNN, OCR_CRNN
from src.engine import train_epoch, evaluate

def get_args():
    parser = argparse.ArgumentParser(description="Train OCR Model")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["crnn", "simplecnn"], 
        default="crnn",
        help="Choose model architecture: 'crnn' or 'simplecnn'"
    )
    return parser.parse_args()

def main():
    args = get_args()
    model_name = args.model
    
    CONFIG['log_file'] = CONFIG['log_file'].parent / f"training_log_{model_name}.txt"
    CONFIG['csv_file'] = CONFIG['csv_file'].parent / f"training_metrics_{model_name}.csv"
    CONFIG['save_path'] = CONFIG['save_path'].parent / f"best_{model_name}.pth"
    CONFIG['log_dir'] = CONFIG['log_dir'].parent / f"runs_{model_name}"
    
    logger = setup_logging(CONFIG['log_file'])
    logger.info(f"### Starting Training for Model: {model_name.upper()}")
    logger.info(f"### Configuration loaded. Device: {CONFIG['device']}")
    logger.info(f"### Logs will be saved to: {CONFIG['log_file']}")

    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_h'], CONFIG['image_w'])), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])
    
    ds = UkrainianOCRDataset(CONFIG['metafile'], CONFIG['data_dir'], CONFIG['alphabet'], transform)
    
    total_len = len(ds)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    train_ds, val_ds, test_ds = random_split(ds, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)

    if not os.path.exists(CONFIG['csv_file']):
        with open(CONFIG['csv_file'], mode='w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_CER", "Val_WER", "Learning_Rate", "Epoch_Time_Sec"])
        logger.info(f"### Created new metrics CSV: {CONFIG['csv_file']}")
    else:
        logger.info(f"### Found existing metrics CSV, appending new data.")

    logger.info(f"### Initializing {model_name.upper()} Model...")
    if model_name == "crnn":
        model = OCR_CRNN(ds.vocab_size).to(CONFIG['device'])
    else:
        model = OCR_SimpleCNN(ds.vocab_size).to(CONFIG['device'])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"### Total Parameters: {total_params:,}")
    logger.info(f"### Trainable Parameters: {trainable_params:,}")
    
    # Overfitting 1 Batch 
    logger.info("\n### STARTING SANITY CHECK: Overfitting 1 Batch...")
    
    batch = next(iter(train_loader)) 
    images, targets, target_lengths = batch
    
    images = images.to(CONFIG['device'])
    targets = targets.to(CONFIG['device'])
    target_lengths = target_lengths.to(CONFIG['device'])
    
    if model_name == "crnn":
        sanity_model = OCR_CRNN(ds.vocab_size).to(CONFIG['device'])
    else:
        sanity_model = OCR_SimpleCNN(ds.vocab_size).to(CONFIG['device'])
        
    sanity_optim = optim.Adam(sanity_model.parameters(), lr=0.001)
    sanity_criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    for k in range(50):
        sanity_optim.zero_grad()
        preds = sanity_model(images)
        input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(1), dtype=torch.long)
        
        loss = sanity_criterion(preds.permute(1, 0, 2), targets, input_lengths, target_lengths)
        loss.backward()
        sanity_optim.step()
        
        if k % 10 == 0:
            logger.info(f"   Sanity Epoch {k}/50 | Loss: {loss.item():.4f}")
            
    logger.info(f"### Sanity Check Complete (Final Loss: {loss.item():.4f})\n")
  

    CONFIG['save_path'].parent.mkdir(parents=True, exist_ok=True)
    
    LATEST_PATH = f"checkpoints/latest_{model_name}.pth"
    
    if os.path.exists(LATEST_PATH):
        logger.info(f"### Found checkpoint '{LATEST_PATH}'. Loading...")
        try:
            model.load_state_dict(torch.load(LATEST_PATH, map_location=CONFIG['device']))
            logger.info("### Weights loaded successfully! Resuming training...")
        except:
            logger.info("### Error loading weights (Architecture mismatch?). Starting from scratch.")
    else:
        logger.info("### No checkpoint found. Starting training from scratch.")
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    writer = SummaryWriter(CONFIG['log_dir'])
    best_cer = float('inf')
    
    for epoch in range(CONFIG['num_epochs']):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, ds.idx2char, epoch, writer, CONFIG)
        val_loss, val_cer, val_wer = evaluate(model, val_loader, criterion, ds.idx2char, CONFIG)
        
        duration = time.time() - start_time
        
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/CER", val_cer, epoch)
        writer.add_scalar("Val/WER", val_wer, epoch)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        with open(CONFIG['csv_file'], mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                epoch + 1, 
                f"{train_loss:.4f}", 
                f"{val_loss:.4f}", 
                f"{val_cer:.4f}", 
                f"{val_wer:.4f}", 
                f"{current_lr:.6f}", 
                f"{duration:.2f}"
            ])
        logger.info(f"### Metrics saved to {CONFIG['csv_file']}")

        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), CONFIG['save_path'])
            logger.info(f"### Best Model Saved to {CONFIG['save_path']}! (CER: {best_cer:.4f})")
        
        torch.save(model.state_dict(), LATEST_PATH)
    
    writer.close()
    logger.info("### Training Complete.")

    logger.info("\n### RELOADING BEST MODEL FOR TEST EVALUATION...")
    if CONFIG['save_path'].exists():
        model.load_state_dict(torch.load(CONFIG['save_path'], map_location=CONFIG['device']))
    
    test_loss, test_cer, test_wer = evaluate(model, test_loader, criterion, ds.idx2char, CONFIG)
    
    logger.info("-" * 50)
    logger.info(f"FINAL TEST RESULTS ({model_name}) | Loss: {test_loss:.4f} | CER: {test_cer:.4f} | WER: {test_wer:.4f}")
    logger.info("-" * 50)

if __name__ == "__main__":
    main()