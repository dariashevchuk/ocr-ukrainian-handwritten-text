import torch
import torch.nn as nn
from src.model import OCR_CRNN, OCR_SimpleCNN
from torchinfo import summary
import onnx

device = "cpu"
vocab_size = 50 
h, w = 64, 800
dummy_input = torch.randn(1, 1, h, w).to(device)

def analyze_and_export(model_class, name):
    print(f"\n{'='*20} ANALYZING: {name} {'='*20}")
    
    model = model_class(vocab_size).to(device)
    model.eval()

    try:
        model_stats = summary(model, input_size=(1, 1, h, w), 
                              col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                              verbose=0)
        print(model_stats)
        print(f"\n>>> Total Params: {model_stats.total_params:,}")
        print(f">>> Trainable Params: {model_stats.trainable_params:,}")
    except Exception as e:
        print(f"Error generating statistics: {e}")


    onnx_path = f"{name}_viz.onnx"
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path, 
            verbose=False,
            input_names=['Input Image (64x800)'],
            output_names=['Sequence Output'],
            opset_version=14 
        )
        print(f"\n>>> SUCCESS: Saved diagram to {onnx_path}")
        print(">>> ACTION: Upload this .onnx file to https://netron.app for your report screenshots.")
    except Exception as e:
        print(f"\n>>> WARNING: Diagram export failed ({e})")
        print(">>> fallback: Use the printed table above for your 'Model Analysis' section.")

if __name__ == "__main__":
    analyze_and_export(OCR_SimpleCNN, "SimpleCNN")
    analyze_and_export(OCR_CRNN, "CRNN_ResNet18")
