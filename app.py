import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from src.config import CONFIG
from src.model import OCR_SimpleCNN, OCR_CRNN
from src.utils import decode_prediction

# 1. Setup Resources
# Read alphabet
with open(CONFIG['alphabet'], 'r', encoding='utf-8') as f:
    lines = f.readlines()
    alphabet = "".join([line.strip('\n').strip('\r') for line in lines])
if ' ' not in alphabet: alphabet += " "

idx2char = {idx + 1: char for idx, char in enumerate(alphabet)}
vocab_size = len(alphabet)

# Global variables to cache the model so we don't reload it unnecessarily
current_model = None
current_model_name = None

# 2. Define Transform
transform = transforms.Compose([
    transforms.Resize((CONFIG['image_h'], CONFIG['image_w'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 3. Model Loading Helper
def load_model(model_choice):
    global current_model, current_model_name
    
    # If the requested model is already loaded, skip reloading
    if current_model is not None and current_model_name == model_choice:
        return current_model

    print(f"Loading {model_choice} model...")
    device = CONFIG['device']
    
    if model_choice == "CRNN (ResNet)":
        model = OCR_CRNN(vocab_size=vocab_size).to(device)
        # Path determined by train.py logic: checkpoints/best_crnn.pth
        ckpt_path = CONFIG['save_path'].parent / "best_crnn.pth" 
    else:
        model = OCR_SimpleCNN(vocab_size=vocab_size).to(device)
        # Path determined by train.py logic: checkpoints/best_simplecnn.pth
        ckpt_path = CONFIG['save_path'].parent / "best_simplecnn.pth"

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Successfully loaded weights from {ckpt_path}")
    else:
        print(f"WARNING: Checkpoint {ckpt_path} not found. Using random weights.")
    
    model.eval()
    
    # Update cache
    current_model = model
    current_model_name = model_choice
    return current_model

# 4. Prediction Function
def predict_text(image, model_choice):
    if image is None: return "No image provided"
    
    # Ensure correct model is loaded
    model = load_model(model_choice)
    
    # Preprocess
    image = image.convert("L") 
    image_tensor = transform(image).unsqueeze(0).to(CONFIG['device']) 
    
    # Inference
    with torch.no_grad():
        preds = model(image_tensor)
        decoded_text = decode_prediction(preds[0], idx2char)
    
    return decoded_text

# 5. Launch Interface
if __name__ == "__main__":
    # Define available models
    model_options = ["CRNN (ResNet)", "SimpleCNN"]
    
    demo = gr.Interface(
        fn=predict_text,
        inputs=[
            gr.Image(type="pil", label="Upload Handwriting"),
            gr.Dropdown(choices=model_options, value="CRNN (ResNet)", label="Choose Model")
        ],
        outputs=gr.Textbox(label="Predicted Text"),
        title="Ukrainian OCR Demo",
        description="Select a model and upload an image to recognize text."
    )
    demo.launch()