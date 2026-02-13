# Project 3: Handwritten Text Recognition (OCR)

## Check out the full report: [link](https://drive.google.com/file/d/1HoJyBlUIDVkOIFVFfwaYn1Ct5odT4k5D/view?usp=sharing)

This project implements an Optical Character Recognition (OCR) system for handwritten text using PyTorch. It features two distinct neural network architectures, a complete training pipeline with logging, and a web-based GUI for inference. Dataset used:  https://www.kaggle.com/datasets/annyhnatiuk/ukrainian-handwritten-text

## Project Overview

* Task: Sequence-to-Sequence recognition (Image to Text).
* Input: Grayscale images of handwriting (normalized).
* Loss Function: CTCLoss (Connectionist Temporal Classification).
* Tools: PyTorch, Tensorboard, Gradio, Docker.


## Models Implemented
We compare two architectures in this project:

CRNN (ResNet18): Uses a pre-trained ResNet18 (Transfer Learning) as the feature extractor.

SimpleCNN: A custom, lightweight 5-layer CNN architecture designed from scratch (>50% own layers).

Both models use a Bidirectional LSTM for sequence modeling.

## Getting Started
You can run this project using Docker (recommended) or a local Python environment.

1. Option A: Using Docker (Easiest)
This project is containerized for easy deployment (CPU-optimized).

- Build the image:
docker-compose build

- Run the Web GUI:
docker-compose up app
Access the interface at http://localhost:7860.

- Run Training:
docker-compose up train

2. Option B: Local Installation

Install dependencies:
pip install -r requirements.txt
Run the Web GUI:

python3 app.py


## Training
To train a model, run the train.py script. You can specify which architecture to use via arguments.

Train the Custom Model:
python train.py --model simplecnn

Train the Transfer Learning Model:
python train.py --model crnn

Training Features:
Automatic Logging: Metrics (Loss, CER, WER) are saved to logs/ (CSV & Tensorboard).
Sanity Check: Runs a quick overfitting test on a single batch before training starts.
Adaptive Learning Rate: Uses ReduceLROnPlateau scheduler.
Checkpoints: Best models are automatically saved to checkpoints/.

## Evaluation Metrics
The models are evaluated using:
CER (Character Error Rate): Levenshtein distance between predicted and target characters.
WER (Word Error Rate): Percentage of incorrectly predicted words.

## Architecture Comparison

| Metric          | SimpleCNN (Custom) | CRNN (ResNet18) |
|----------------|--------------------:|----------------:|
| Parameters     | 4.2 Million         | 12.8 Million    |
| Weight Size    | ~16.8 MB            | ~51.1 MB        |
| Compute Cost   | 4.34 GMacs          | 13.36 GMacs     |
| Output Sequence| 200 steps           | 400 steps       |

## Results and Evaluation

| Feature          | SimpleCNN (Model B) | CRNN ResNet18 (Model A) |
|-----------------|--------------------:|------------------------:|
| Best Val CER    | 18.51%              | 10.90% (Winner)         |
| Best Val WER    | 61.74%              | 42.21%                  |
| Training Time   | ~1.6h               | ~2.8h                   |
















