


1. Prepare the data 
size of each image should be 1740 * 200. add white pixels to each image so that the text doesnt change its original size

2. Normilize the color to [0, 1] range, float32

3. add some distortions to some images for bonus points

3. Read about the models and understand how to implement them



### The simplest concrete checklist (do this, you’ll be safe)

1) Prepare data
Pad images to 1740×200 with white pixels (no distortion)
Encode labels with your alphabet file
Train/val/test split

2) Train baseline

CRNN (CNN → BiLSTM → linear) + CTC
Decode with greedy CTC
Report CER/WER

3) Train your variant

Custom CNN blocks (your design) + BiLSTM + CTC
Show that it improves (or at least compare honestly)

4) Add the 3 easy extras
augmentation
mall hyperparam search
overfit-10-samples experiment

That’s the most beginner-friendly path to ≥8 points with Model ≥3.


---------------------------------------------------------------------------------------------------------------

1. I have the dataset ready and padded      DONE

2. Comparison of 3 models:
- ResNet18 + BiLSTM (CRNN)   
- ResNet18 (CNN-Only)
- TrOCR transformer 

3. Run the first one with more epochs 
- Create checkpoints to save latest models
- Training CER (Character Error Rate). Implementation: Calculate CER on the current batch every time you log.
- Gradient Norm. Implementation: You are already clipping it (clip_grad_norm_), just calculate the value before clipping.
- Learning Rate. Since you use ReduceLROnPlateau, the learning rate will change. Plotting this proves your scheduler is working.

4. train the Resnet18 alone, output the metrics. 
5. Use TensorBoard

6. Include overfitting 
-------------------------------------------------------------------------------------------------------------

1. Split data into train, validation and test set


===============================================================================================================
 1) Problem 2p 
 2) Model CRNN transfer learing 1p + my own simpleCRNN 2p 
 3) Adaptive hyperparameters LR 1p 
 4) Overfitting 1p 
 5) Tensorboard 1p 
 6) Docker 1p 
 7) GUI Gradio 1p 

Problem >= 1pk
Model >= 2pk
Sum of additional points from dataset, training, tools, report >= 3pk
Sum of points >=10 (pair) 9 (alone)
