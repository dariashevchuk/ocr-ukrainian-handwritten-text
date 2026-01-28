import torch
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import logging
from src.config import CONFIG

logger = logging.getLogger()

class UkrainianOCRDataset(Dataset):
    def __init__(self, metafile_path, root_dir, alphabet_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(metafile_path, sep='\t', names=['filename', 'label'], dtype=str).dropna()
        
        with open(alphabet_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.alphabet = "".join([line.strip('\n').strip('\r') for line in lines])

        if ' ' not in self.alphabet:
            logger.warning("### Space not found in file. Adding manually.")
            self.alphabet += " "

        logger.info(f"### Alphabet Loaded. Size: {len(self.alphabet)}")
        
        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.alphabet)}
        self.idx2char = {idx + 1: char for idx, char in enumerate(self.alphabet)}
        self.vocab_size = len(self.alphabet)
        
    def __len__(self):
        return len(self.df)

    def text_to_labels(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        text_label = row['label']
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("L")
            if self.transform: image = self.transform(image)
        except:
            # Uses CONFIG here as in original script
            image = torch.zeros(1, CONFIG['image_h'], CONFIG['image_w'])
            
        encoded_label = torch.tensor(self.text_to_labels(text_label), dtype=torch.long)
        return image, encoded_label, len(encoded_label)

def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(labels)
    target_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return images, targets, target_lengths