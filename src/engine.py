import torch
import jiwer
import logging
import random
from src.utils import decode_prediction, decode_target

logger = logging.getLogger()

def train_epoch(model, loader, criterion, optimizer, idx2char, epoch, writer, config):
    model.train()
    total_loss = 0
    
    logger.info(f"\n--- Epoch {epoch+1} Training ---")
    for i, (images, targets, target_lengths) in enumerate(loader):
        images = images.to(config['device'])
        targets = targets.to(config['device'])
        target_lengths = target_lengths.to(config['device'])
        
        optimizer.zero_grad()
        preds = model(images)
        
        input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(1), dtype=torch.long)
        loss = criterion(preds.permute(1, 0, 2), targets, input_lengths, target_lengths)
        
        loss.backward()
        
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        total_loss += loss.item()
        
        if i % config['log_interval'] == 0:
            global_step = epoch * len(loader) + i
            
            with torch.no_grad():
                pred_str = decode_prediction(preds[0], idx2char)
                tgt_len = target_lengths[0].item()
                tgt_str = decode_target(targets[:tgt_len], idx2char)
                
                try:
                    batch_cer = jiwer.cer(tgt_str, pred_str)
                except:
                    batch_cer = 1.0 

                current_lr = optimizer.param_groups[0]['lr']
                
                logger.info(f"Batch {i}/{len(loader)} | Loss: {loss.item():.4f} | CER: {batch_cer:.4f} | LR: {current_lr:.6f} | Grad: {grad_norm:.4f}")
                logger.info(f"   GT:   {tgt_str}")
                logger.info(f"   Pred: {pred_str}")
                logger.info("-" * 50)

                if writer:
                    writer.add_scalar("Train/Loss", loss.item(), global_step)
                    writer.add_scalar("Train/CER", batch_cer, global_step)
                    writer.add_scalar("Train/LearningRate", current_lr, global_step)
                    writer.add_scalar("Train/GradientNorm", grad_norm, global_step)

    return total_loss / len(loader)

def evaluate(model, loader, criterion, idx2char, config):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(config['device'])
            targets = targets.to(config['device'])
            target_lengths = target_lengths.to(config['device'])
            
            preds = model(images)
            input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(1), dtype=torch.long)
            loss = criterion(preds.permute(1, 0, 2), targets, input_lengths, target_lengths)
            total_loss += loss.item()
            
            start_ptr = 0
            for k in range(len(images)):
                pred_str = decode_prediction(preds[k], idx2char)
                tgt_len = target_lengths[k].item()
                tgt_str = decode_target(targets[start_ptr : start_ptr + tgt_len], idx2char)
                start_ptr += tgt_len
                
                all_preds.append(pred_str)
                all_targets.append(tgt_str)

    cer = jiwer.cer(all_targets, all_preds)
    wer = jiwer.wer(all_targets, all_preds)
    
    logger.info("\n" + "="*60)
    logger.info(f"### VALIDATION RESULTS (CER: {cer:.4f} | WER: {wer:.4f})")
    logger.info(f"{'GROUND TRUTH':<40} | {'PREDICTION'}")
    logger.info("-" * 60)
    for idx in random.sample(range(len(all_preds)), min(3, len(all_preds))):
        logger.info(f"{all_targets[idx][:38]:<40} | {all_preds[idx]}")
    logger.info("="*60 + "\n")
    
    return total_loss / len(loader), cer, wer